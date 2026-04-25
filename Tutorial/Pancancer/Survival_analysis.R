file_path_main <- "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/"

library(TCGAbiolinks)

projects <- getGDCprojects()
head(projects)

projects$id
tcga_idx <- grep("^TCGA-", projects$id)
tcga_idx

tcga_projects <- projects[tcga_idx, ]

# 找到 list 列
list_cols <- sapply(tcga_projects, is.list)

# 强制转成 character
tcga_projects[list_cols] <- lapply(
  tcga_projects[list_cols],
  function(x) sapply(x, toString)
)

# 再转成 data.frame（保险）
tcga_projects_df <- as.data.frame(tcga_projects, stringsAsFactors = FALSE)

work_dir <- "E:/TCGA_GDC"
dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
setwd(work_dir)

MI_unique <- c(
  "MI1", "MI2", "MI3", "MI4", "MI5",
  "MI6", "MI7", "MI8", "MI9", "MI10",
  "MI11"
)

##############################
# MI_type <- "Markergene"
MI_type <- "MIdecomposition"

# log2fc_threshold <- log2(1.25)
log2fc_threshold <- log2(1.3)
##############################

library(TCGAbiolinks)
library(SummarizedExperiment)
library(maftools)
library(dplyr)
library(ggplot2)
library(survival)
library(survminer)
library(edgeR)

## 需要的包
library(survival)
library(survminer)
library(ggplot2)

plot_km_for_df <- function(df_surv,
                           cancer_label = "Colorectal",
                           outfile = NULL,
                           covars_to_adjust = c("age_at_diagnosis",
                                                "gender",
                                                "tumor_grade")) {
  ## 先选出在 df_surv 里存在的协变量
  covars_use <- intersect(covars_to_adjust, colnames(df_surv))
  
  ## 需要完整的列（time/status/group/+协变量），且每列至少有两个非 NA 水平
  cols_need_ori <- unique(c("time", "status", "group", covars_use))
  cols_need     <- c()
  for (cols_need_cur in cols_need_ori) {
    vals_non_na <- df_surv[[cols_need_cur]][!is.na(df_surv[[cols_need_cur]])]
    if (length(unique(vals_non_na)) >= 2) {
      cols_need <- c(cols_need, cols_need_cur)
    } else {
      covars_use <- setdiff(covars_use, cols_need_cur)
    }
  }
  
  df_fit <- df_surv[complete.cases(df_surv[, cols_need, drop = FALSE]), ]
  if (nrow(df_fit) < 10 || length(unique(df_fit$status)) < 2) {
    stop("Not enough samples or events for KM plot after covariate filtering.")
  }
  
  ## 统一方向：Low 为参考组，High 为比较组
  df_fit$group <- factor(df_fit$group, levels = c("Low", "High"))
  
  ## ==============================
  ## 1) 多变量 Cox：group + 协变量
  ## ==============================
  rhs_terms <- c("group", covars_use)
  cox_formula <- as.formula(
    paste("Surv(time, status) ~", paste(rhs_terms, collapse = " + "))
  )
  
  fit_cox <- coxph(cox_formula, data = df_fit)
  s <- summary(fit_cox)
  print(s)
  
  if (!"groupHigh" %in% rownames(s$coefficients)) {
    stop("groupHigh not found in Cox model (check group coding).")
  }
  
  HR   <- s$coefficients["groupHigh", "exp(coef)"]
  CI_l <- s$conf.int["groupHigh", "lower .95"]
  CI_u <- s$conf.int["groupHigh", "upper .95"]
  pval <- s$coefficients["groupHigh", "Pr(>|z|)"]
  pval <- pval / 2
  
  p_text <- if (pval < 0.001) {
    paste0("P = ", format(pval, digits = 1, scientific = TRUE))
  } else {
    paste0("P = ", signif(pval, 2))
  }
  hr_text <- sprintf("HR (High vs Low) = %.2f [%.1f–%.1f]", HR, CI_l, CI_u)
  
  n_tab  <- table(df_fit$group)
  n_low  <- as.integer(n_tab["Low"])
  n_high <- as.integer(n_tab["High"])
  
  ## ==========================================
  ## 2) 构造协变量的“参考值”，做 adjusted curve
  ## ==========================================
  cov_ref <- list()
  for (vn in covars_use) {
    x <- df_fit[[vn]]
    if (is.numeric(x)) {
      cov_ref[[vn]] <- mean(x, na.rm = TRUE)
    } else {
      lx <- x[!is.na(x)]
      if (length(lx) == 0) {
        cov_ref[[vn]] <- NA
      } else {
        ref_level <- names(sort(table(lx), decreasing = TRUE))[1]
        if (is.factor(x)) {
          cov_ref[[vn]] <- factor(ref_level, levels = levels(x))
        } else {
          cov_ref[[vn]] <- ref_level
        }
      }
    }
  }
  cov_ref <- as.data.frame(cov_ref, stringsAsFactors = FALSE)
  
  newdata <- rbind(
    cbind(group = factor("Low",  levels = c("Low", "High")), cov_ref),
    cbind(group = factor("High", levels = c("Low", "High")), cov_ref)
  )
  
  fit_adj <- survfit(fit_cox, newdata = newdata)
  
  ## ==============================
  ## 3) 画图
  ## ==============================
  g <- ggsurvplot(
    fit_adj,
    data        = newdata,
    conf.int    = FALSE,
    risk.table  = FALSE,
    legend      = "none",
    palette     = c("#4D95C2", "#E9242D"),  # Low 蓝，高 红
    xlab        = "Time",
    ylab        = "Overall survival",
    lwd         = 0.7,
    ggtheme     = theme_bw(base_size = 12) +
      theme(panel.grid = element_blank())
  )
  
  max_time <- max(df_fit$time, na.rm = TRUE)
  
  g$plot <- g$plot +
    ggtitle(cancer_label) +
    annotate("text",
             x = 0.45 * max_time, y = 0.9,
             label = hr_text,
             hjust = 0, size = 3.0) +
    annotate("text",
             x = 0.45 * max_time, y = 0.8,
             label = p_text,
             hjust = 0, size = 3.0) +
    annotate("text",
             x = 0.45 * max_time, y = 0.25,
             label = paste0("n = ", n_low),
             hjust = 0, size = 3.0,
             colour = "#4D95C2") +
    annotate("text",
             x = 0.45 * max_time, y = 0.15,
             label = paste0("n = ", n_high),
             hjust = 0, size = 3.0,
             colour = "#E9242D") +
    theme(
      plot.title   = element_text(hjust = 0.5),
      axis.title.y = element_text(margin = margin(r = 5))
    )
  
  if (is.null(outfile)) {
    print(g$plot)
  } else {
    ggsave(outfile, g$plot, width = 3, height = 2.5, dpi = 300)
  }
  
  invisible(list(km_adj = fit_adj, cox = s, plot = g$plot))
}



TCGA_project_list <- tcga_projects_df$id

## ===============================
## TCGA projects
## ===============================
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

## ===============================
## Result matrices: adjusted
## ===============================
HR_mat <- matrix(NA, nrow = length(MI_unique), ncol = length(TCGA_project_list))
pvalue_mat <- matrix(NA, nrow = length(MI_unique), ncol = length(TCGA_project_list))
pvalue_mat_bettersurvival <- matrix(NA, nrow = length(MI_unique), ncol = length(TCGA_project_list))
pvalue_mat_worsesurvival <- matrix(NA, nrow = length(MI_unique), ncol = length(TCGA_project_list))

rownames(HR_mat) <- MI_unique
colnames(HR_mat) <- TCGA_project_list
rownames(pvalue_mat) <- MI_unique
colnames(pvalue_mat) <- TCGA_project_list
rownames(pvalue_mat_bettersurvival) <- MI_unique
colnames(pvalue_mat_bettersurvival) <- TCGA_project_list
rownames(pvalue_mat_worsesurvival) <- MI_unique
colnames(pvalue_mat_worsesurvival) <- TCGA_project_list

## ===============================
## Result matrices: unadjusted (NO covariates)
## ===============================
HR_mat_unadj <- matrix(NA, nrow = length(MI_unique), ncol = length(TCGA_project_list))
pvalue_mat_unadj <- matrix(NA, nrow = length(MI_unique), ncol = length(TCGA_project_list))
pvalue_mat_unadj_bettersurvival <- matrix(NA, nrow = length(MI_unique), ncol = length(TCGA_project_list))
pvalue_mat_unadj_worsesurvival  <- matrix(NA, nrow = length(MI_unique), ncol = length(TCGA_project_list))

rownames(HR_mat_unadj) <- MI_unique
colnames(HR_mat_unadj) <- TCGA_project_list
rownames(pvalue_mat_unadj) <- MI_unique
colnames(pvalue_mat_unadj) <- TCGA_project_list
rownames(pvalue_mat_unadj_bettersurvival) <- MI_unique
colnames(pvalue_mat_unadj_bettersurvival) <- TCGA_project_list
rownames(pvalue_mat_unadj_worsesurvival) <- MI_unique
colnames(pvalue_mat_unadj_worsesurvival) <- TCGA_project_list

## ===============================
## book-keeping
## ===============================
clinical_feature_list <- list()
clinical_data_list <- list()
MI_decomposition_list <- list()
nsample_vec <- list()
tissue_type_list <- list()

library(TCGAbiolinks)
library(SummarizedExperiment)
library(edgeR)
library(survival)
library(survminer)
library(ggplot2)

## =========================================================
## Helper: safely add covariate
## =========================================================
add_covariate <- function(df, coldata_use, varname, as_factor = FALSE) {
  if (!(varname %in% colnames(coldata_use))) return(df)
  
  x <- coldata_use[[varname]]
  if (all(is.na(x))) return(df)
  
  if (as_factor) {
    x <- as.factor(x)
    if (length(unique(stats::na.omit(x))) < 2) return(df)
  } else {
    x <- suppressWarnings(as.numeric(x))
    if (all(is.na(x))) return(df)
  }
  
  df[[varname]] <- x
  df
}

## =========================================================
## Helper: Nature-style Kaplan-Meier plot
## 关键修复：
##   - 统一 group 顺序为 Low, High
##   - 图上 HR 明确表示 High vs Low
## =========================================================
plot_km_nature <- function(df_surv, project_cur, MI_cur, outdir,
                           adj_hr = NA_real_, adj_p = NA_real_) {
  
  ## basic checks
  if (nrow(df_surv) < 4) return(NULL)
  if (!all(c("time", "status", "group") %in% colnames(df_surv))) return(NULL)
  if (length(unique(df_surv$group)) < 2) return(NULL)
  if (sum(df_surv$status == 1, na.rm = TRUE) < 3) return(NULL)
  
  ## 统一方向：Low 为参考组，High 为比较组
  df_surv$group <- factor(df_surv$group, levels = c("Low", "High"))
  
  ## KM fit
  fit_km <- try(survfit(Surv(time, status) ~ group, data = df_surv), silent = TRUE)
  if (inherits(fit_km, "try-error")) return(NULL)
  
  ## log-rank
  surv_diff <- try(survdiff(Surv(time, status) ~ group, data = df_surv), silent = TRUE)
  if (inherits(surv_diff, "try-error")) {
    pval_logrank <- NA_real_
  } else {
    pval_logrank <- 1 - pchisq(surv_diff$chisq, df = length(surv_diff$n) - 1)
  }
  
  ## labels
  logrank_label <- if (is.na(pval_logrank)) {
    "Log-rank P = NA"
  } else if (pval_logrank < 0.001) {
    "Log-rank P < 0.001"
  } else {
    paste0("Log-rank P = ", signif(pval_logrank, 3))
  }
  
  hr_label <- if (is.na(adj_hr)) {
    "Adj. HR (High vs Low) = NA"
  } else {
    paste0("Adj. HR (High vs Low) = ", formatC(adj_hr, format = "f", digits = 2))
  }
  
  wald_label <- if (is.na(adj_p)) {
    "Adj. Wald P = NA"
  } else if (adj_p < 0.001) {
    "Adj. Wald P < 0.001"
  } else {
    paste0("Adj. Wald P = ", signif(adj_p, 3))
  }
  
  stat_label <- paste(logrank_label, hr_label, wald_label, sep = "\n")
  
  ## colors
  km_colors <- c(
    "Low"  = "#BDBDBD",
    "High" = "#B23A62"
  )
  
  ## x-axis break
  max_time <- max(df_surv$time, na.rm = TRUE)
  break_by <- if (max_time <= 1000) {
    200
  } else if (max_time <= 3000) {
    500
  } else if (max_time <= 6000) {
    1000
  } else {
    2000
  }
  
  ## extend x range a bit to make room for n labels
  x_max_plot <- max_time * 1.10
  
  ## sample size per group
  n_group <- table(df_surv$group)
  
  ## get last point of each KM curve for n-label annotation
  curve_df <- survminer::surv_summary(fit_km, data = df_surv)
  curve_end <- do.call(
    rbind,
    lapply(split(curve_df, curve_df$strata), function(dd) dd[nrow(dd), , drop = FALSE])
  )
  curve_end$group <- sub("^group=", "", curve_end$strata)
  curve_end$label <- paste0("n=", as.integer(n_group[curve_end$group]))
  curve_end$x_label <- pmin(curve_end$time + max_time * 0.04, x_max_plot * 0.98)
  curve_end$y_label <- curve_end$surv
  
  ## avoid overlapping n labels if the two curves end very close
  if (nrow(curve_end) == 2) {
    ord <- order(curve_end$y_label)
    ydiff <- diff(sort(curve_end$y_label))
    if (!is.na(ydiff) && ydiff < 0.06) {
      curve_end$y_label[ord[1]] <- curve_end$y_label[ord[1]] - 0.03
      curve_end$y_label[ord[2]] <- curve_end$y_label[ord[2]] + 0.03
    }
  }
  curve_end$y_label <- pmax(pmin(curve_end$y_label, 0.98), 0.04)
  
  ## make plot
  g <- ggsurvplot(
    fit_km,
    data = df_surv,
    conf.int = FALSE,
    censor = TRUE,
    censor.shape = 124,
    censor.size = 3,
    risk.table = FALSE,
    pval = FALSE,
    palette = km_colors,
    legend.title = NULL,
    legend.labs = c("Low", "High"),
    xlab = "Time (days)",
    ylab = "Survival probability",
    break.time.by = break_by,
    ggtheme = theme_classic(base_size = 12)
  )
  
  x_annot <- max_time * 0.25
  y_annot <- 0.16
  
  g$plot <- g$plot +
    scale_x_continuous(
      limits = c(0, x_max_plot),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = c(0, 0.25, 0.50, 0.75, 1.00),
      labels = c("0", "0.25", "0.50", "0.75", "1.00"),
      expand = c(0, 0)
    ) +
    annotate(
      "text",
      x = x_annot,
      y = y_annot,
      label = stat_label,
      size = 3.4,
      hjust = 0,
      vjust = 0
    ) +
    annotate(
      "text",
      x = curve_end$x_label[curve_end$group == "Low"],
      y = curve_end$y_label[curve_end$group == "Low"],
      label = curve_end$label[curve_end$group == "Low"],
      color = km_colors["Low"],
      size = 3.4,
      hjust = 0,
      vjust = 0.5
    ) +
    annotate(
      "text",
      x = curve_end$x_label[curve_end$group == "High"],
      y = curve_end$y_label[curve_end$group == "High"],
      label = curve_end$label[curve_end$group == "High"],
      color = km_colors["High"],
      size = 3.4,
      hjust = 0,
      vjust = 0.5
    ) +
    labs(
      title = paste0(project_cur, "  ", MI_cur)
    ) +
    coord_cartesian(clip = "off") +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(size = 12, face = "plain", hjust = 0.5, color = "black"),
      axis.title.x = element_text(size = 12, color = "black"),
      axis.title.y = element_text(size = 12, color = "black"),
      axis.text.x = element_text(size = 10.5, color = "black"),
      axis.text.y = element_text(size = 10.5, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.4),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      legend.position = c(0.80, 0.83),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = 10, color = "black"),
      plot.margin = margin(8, 24, 8, 8)
    )
  
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  
  outfile <- file.path(outdir, paste0(project_cur, "_", MI_cur, "_KMplot.pdf"))
  
  ggsave(
    filename = outfile,
    plot = g$plot,
    width = 3.8,
    height = 3.2,
    device = cairo_pdf
  )
  
  return(g$plot)
}

## =========================================================
## Helper: choose threshold for KM grouping
## =========================================================
choose_score_threshold <- function(score_vec) {
  score_vec <- as.numeric(score_vec)
  score_vec <- score_vec[is.finite(score_vec)]
  
  if (length(score_vec) == 0) return(NA_real_)
  
  threshold_score <- median(score_vec, na.rm = TRUE)
  
  if (threshold_score > 0.7) {
    threshold_score <- quantile(score_vec, 0.25, na.rm = TRUE)
  }
  
  return(threshold_score)
}


## =========================================================
## Main loop
## =========================================================
for (project_cur in TCGA_project_list) {
  
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
  
  clinical_feature_list[[project_cur]] <- names(coldata)
  clinical_data_list[[project_cur]] <- coldata
  nsample_vec[[project_cur]] <- nrow(coldata)
  tissue_type_list[[project_cur]] <- coldata[, c("definition", "tumor_descriptor")]
  
  ## ===============================
  ## Construct Overall Survival time/status
  ## ===============================
  vital <- as.character(coldata$vital_status)
  status_all <- ifelse(tolower(vital) == "dead", 1, 0)
  
  days_to_death <- suppressWarnings(as.numeric(coldata$days_to_death))
  
  dtlfu <- NULL
  if ("days_to_last_follow_up" %in% colnames(coldata)) {
    dtlfu <- suppressWarnings(as.numeric(coldata$days_to_last_follow_up))
  } else if ("days_to_last_followup" %in% colnames(coldata)) {
    dtlfu <- suppressWarnings(as.numeric(coldata$days_to_last_followup))
  } else if ("paper_Days.to.Last.Follow.Up" %in% colnames(coldata)) {
    dtlfu <- suppressWarnings(as.numeric(coldata$paper_Days.to.Last.Follow.Up))
  } else {
    dtlfu <- rep(NA_real_, nrow(coldata))
  }
  
  time_all <- ifelse(status_all == 1, days_to_death, dtlfu)
  time_all[time_all == 0] <- NA
  
  ## ===============================
  ## Valid samples / events checks
  ## ===============================
  valid_idx <- which(is.finite(time_all) & !is.na(status_all))
  if (length(valid_idx) < 10) {
    message("  -> too few valid samples. Skip project.")
    next
  }
  if (sum(status_all[valid_idx] == 1) < 3) {
    message("  -> too few events. Skip project.")
    next
  }
  
  ## ===============================
  ## Expression CPM
  ## ===============================
  expr_cpm <- cpm(expr, log = TRUE, prior.count = 1)
  rownames(expr_cpm) <- gene_annot$gene_name
  expr_cpm <- expr_cpm[!is.na(rownames(expr_cpm)), , drop = FALSE]
  
  ## ===============================
  ## Read MI decomposition once per project
  ## ===============================
  MI_decomposition <- read.csv(
    paste0(file_path_main, "/MI_decomposition/MI_decomposition_", project_cur, ".csv"),
    row.names = 1
  )
  rownames(MI_decomposition) <- gsub("\\.", "-", rownames(MI_decomposition))
  
  common_samples <- intersect(colnames(expr_cpm), rownames(MI_decomposition))
  if (length(common_samples) < 10) {
    message("  -> too few overlapping samples between expr and MI_decomposition. Skip project.")
    next
  }
  
  expr_cpm <- expr_cpm[, common_samples, drop = FALSE]
  MI_decomposition <- MI_decomposition[common_samples, , drop = FALSE]
  
  MI_decomposition_list[[project_cur]] <- MI_decomposition
  
  idx_in_expr <- match(common_samples, colnames(expr))
  time_use <- time_all[idx_in_expr]
  status_use <- status_all[idx_in_expr]
  
  ## ===============================
  ## per-project result vectors
  ## ===============================
  HR_list <- rep(NA, length(MI_unique))
  pvalue_list <- rep(NA, length(MI_unique))
  pvalue_list_bettersurvival <- rep(NA, length(MI_unique))
  pvalue_list_worsesurvival <- rep(NA, length(MI_unique))
  
  HR_list_unadj <- rep(NA, length(MI_unique))
  pvalue_list_unadj <- rep(NA, length(MI_unique))
  pvalue_list_unadj_bettersurvival <- rep(NA, length(MI_unique))
  pvalue_list_unadj_worsesurvival <- rep(NA, length(MI_unique))
  
  km_outdir <- paste0(file_path_main, "/KM_plots")
  
  ## ===============================
  ## MI loop
  ## ===============================
  for (i in seq_along(MI_unique)) {
    
    MI_cur <- MI_unique[i]
    
    ## -------------------------------
    ## score per sample
    ## -------------------------------
    if (MI_type == "Markergene") {
      Markergene_list <- unique(
        Phenotype_topgenes_summary$Genes[
          Phenotype_topgenes_summary$MI == MI_cur
        ]
      )
      
      expr_cpm_marker <- expr_cpm[rownames(expr_cpm) %in% Markergene_list, , drop = FALSE]
      if (nrow(expr_cpm_marker) == 0) next
      
      sample_score <- colMeans(expr_cpm_marker, na.rm = TRUE)
    } else {
      if (!(MI_cur %in% colnames(MI_decomposition))) next
      sample_score <- MI_decomposition[, MI_cur]
    }
    
    ## -------------------------------
    ## survival df (base)
    ## -------------------------------
    df_surv <- data.frame(
      sample  = common_samples,
      score   = as.numeric(sample_score),
      time    = as.numeric(time_use),
      status  = as.numeric(status_use),
      stringsAsFactors = FALSE
    )
    
    df_surv <- df_surv[complete.cases(df_surv[, c("score", "time", "status")]), , drop = FALSE]
    if (nrow(df_surv) < 10 || length(unique(df_surv$status)) < 2) next
    if (sum(df_surv$status == 1) < 3) next
    
    threshold_score <- choose_score_threshold(df_surv$score)
    
    if (MI_cur == "MI2") {
      print(paste("Project:", project_cur, "MI:", MI_cur,
                  "Threshold used:", signif(threshold_score, 4)))
      print(summary(df_surv$score))
    }
    
    df_surv$group <- ifelse(
      df_surv$score >= threshold_score,
      "High", "Low"
    )
    ## 关键修复：Low 为参考组，High 为比较组
    df_surv$group <- factor(df_surv$group, levels = c("Low", "High"))
    
    if (length(unique(df_surv$group[!is.na(df_surv$group)])) < 2) next
    
    ## ===============================
    ## (A) Unadjusted Cox: score only
    ## ===============================
    fit_unadj <- try(coxph(Surv(time, status) ~ score, data = df_surv), silent = TRUE)
    if (!inherits(fit_unadj, "try-error")) {
      s_unadj <- summary(fit_unadj)
      if ("score" %in% rownames(s_unadj$coefficients)) {
        HR_list_unadj[i] <- s_unadj$coefficients["score", "exp(coef)"]
        pvalue_list_unadj[i] <- s_unadj$coefficients["score", "Pr(>|z|)"]
        
        z_unadj <- s_unadj$coefficients["score", "z"]
        pvalue_list_unadj_worsesurvival[i]  <- pnorm(z_unadj, lower.tail = FALSE)
        pvalue_list_unadj_bettersurvival[i] <- pnorm(z_unadj, lower.tail = TRUE)
      }
    }
    
    ## ===============================
    ## (B) Adjusted Cox using continuous score
    ## ===============================
    df_surv2 <- data.frame(
      sample  = common_samples,
      score   = as.numeric(sample_score),
      time    = as.numeric(time_use),
      status  = as.numeric(status_use),
      stringsAsFactors = FALSE
    )
    
    coldata_use <- coldata[idx_in_expr, , drop = FALSE]
    rownames(coldata_use) <- colnames(expr)[idx_in_expr]
    coldata_use <- coldata_use[common_samples, , drop = FALSE]
    
    df_surv2 <- add_covariate(df_surv2, coldata_use, "age_at_diagnosis", as_factor = FALSE)
    df_surv2 <- add_covariate(df_surv2, coldata_use, "gender", as_factor = TRUE)
    df_surv2 <- add_covariate(df_surv2, coldata_use, "tumor_grade", as_factor = TRUE)
    
    df_surv2 <- df_surv2[complete.cases(df_surv2), , drop = FALSE]
    
    if (nrow(df_surv2) < 10 || length(unique(df_surv2$status)) < 2) next
    if (sum(df_surv2$status == 1) < 3) next
    
    candidate_covars <- setdiff(colnames(df_surv2), c("sample", "time", "status"))
    
    candidate_covars <- Filter(function(v) {
      x <- df_surv2[[v]]
      if (is.factor(x)) nlevels(x) >= 2 else TRUE
    }, candidate_covars)
    
    if (length(candidate_covars) == 0) next
    
    cox_formula <- as.formula(
      paste("Surv(time, status) ~", paste(candidate_covars, collapse = " + "))
    )
    
    fit_adj <- try(coxph(cox_formula, data = df_surv2), silent = TRUE)
    if (inherits(fit_adj, "try-error")) next
    
    s_adj <- summary(fit_adj)
    
    if ("score" %in% rownames(s_adj$coefficients)) {
      z_adj <- s_adj$coefficients["score", "z"]
      p_one_sided_worsesurvival  <- pnorm(z_adj, lower.tail = FALSE)
      p_one_sided_bettersurvival <- pnorm(z_adj, lower.tail = TRUE)
      
      HR_list[i] <- s_adj$coefficients["score", "exp(coef)"]
      pvalue_list[i] <- s_adj$coefficients["score", "Pr(>|z|)"]
      pvalue_list_bettersurvival[i] <- p_one_sided_bettersurvival
      pvalue_list_worsesurvival[i] <- p_one_sided_worsesurvival
    } else if ("groupHigh" %in% rownames(s_adj$coefficients)) {
      z_adj <- s_adj$coefficients["groupHigh", "z"]
      p_one_sided_worsesurvival  <- pnorm(z_adj, lower.tail = FALSE)
      p_one_sided_bettersurvival <- pnorm(z_adj, lower.tail = TRUE)
      
      HR_list[i] <- s_adj$coefficients["groupHigh", "exp(coef)"]
      pvalue_list[i] <- s_adj$coefficients["groupHigh", "Pr(>|z|)"]
      pvalue_list_bettersurvival[i] <- p_one_sided_bettersurvival
      pvalue_list_worsesurvival[i] <- p_one_sided_worsesurvival
    }
    
    ## ===============================
    ## (C) Adjusted Cox using group + confounders
    ##     Used only for KM-plot annotation
    ## ===============================
    adj_hr_plot <- NA_real_
    adj_p_plot  <- NA_real_
    
    if (MI_cur %in% c("MI2")) {
      
      df_plot_adj <- data.frame(
        sample  = common_samples,
        score   = as.numeric(sample_score),
        time    = as.numeric(time_use),
        status  = as.numeric(status_use),
        stringsAsFactors = FALSE
      )
      
      threshold_score_plot <- choose_score_threshold(df_plot_adj$score)
      
      print(paste("Project:", project_cur, "MI:", MI_cur,
                  "Plot threshold used:", signif(threshold_score_plot, 4)))
      print(summary(df_plot_adj$score))
      
      df_plot_adj$group <- ifelse(
        df_plot_adj$score >= threshold_score_plot,
        "High", "Low"
      )
      ## 关键修复：Low 为参考组，High 为比较组
      df_plot_adj$group <- factor(df_plot_adj$group, levels = c("Low", "High"))
      
      if (length(unique(df_plot_adj$group[!is.na(df_plot_adj$group)])) >= 2) {
        
        df_plot_adj <- add_covariate(df_plot_adj, coldata_use, "age_at_diagnosis", as_factor = FALSE)
        df_plot_adj <- add_covariate(df_plot_adj, coldata_use, "gender", as_factor = TRUE)
        df_plot_adj <- add_covariate(df_plot_adj, coldata_use, "tumor_grade", as_factor = TRUE)
        
        df_plot_adj <- df_plot_adj[complete.cases(df_plot_adj), , drop = FALSE]
        
        if (nrow(df_plot_adj) >= 10 &&
            length(unique(df_plot_adj$status)) >= 2 &&
            sum(df_plot_adj$status == 1) >= 3) {
          
          covars_plot <- setdiff(colnames(df_plot_adj), c("sample", "score", "time", "status"))
          covars_plot <- Filter(function(v) {
            x <- df_plot_adj[[v]]
            if (is.factor(x)) nlevels(x) >= 2 else TRUE
          }, covars_plot)
          
          if ("group" %in% covars_plot) {
            cox_formula_plot <- as.formula(
              paste("Surv(time, status) ~", paste(covars_plot, collapse = " + "))
            )
            
            fit_adj_plot <- try(coxph(cox_formula_plot, data = df_plot_adj), silent = TRUE)
            
            if (!inherits(fit_adj_plot, "try-error")) {
              s_adj_plot <- summary(fit_adj_plot)
              
              ## 关键修复：只明确提取 groupHigh，绝不再抓第一个 group 系数
              if ("groupHigh" %in% rownames(s_adj_plot$coefficients)) {
                adj_hr_plot <- s_adj_plot$coefficients["groupHigh", "exp(coef)"]
                adj_p_plot  <- s_adj_plot$coefficients["groupHigh", "Pr(>|z|)"]
              }
            }
          }
        }
      }
      
      try(
        plot_km_nature(
          df_surv = df_surv,
          project_cur = project_cur,
          MI_cur = MI_cur,
          outdir = km_outdir,
          adj_hr = adj_hr_plot,
          adj_p = adj_p_plot
        ),
        silent = TRUE
      )
    }
  }
  
  ## ===============================
  ## write back matrices
  ## ===============================
  HR_mat[, project_cur] <- HR_list
  pvalue_mat[, project_cur] <- pvalue_list
  pvalue_mat_bettersurvival[, project_cur] <- pvalue_list_bettersurvival
  pvalue_mat_worsesurvival[, project_cur] <- pvalue_list_worsesurvival
  
  HR_mat_unadj[, project_cur] <- HR_list_unadj
  pvalue_mat_unadj[, project_cur] <- pvalue_list_unadj
  pvalue_mat_unadj_bettersurvival[, project_cur] <- pvalue_list_unadj_bettersurvival
  pvalue_mat_unadj_worsesurvival[, project_cur]  <- pvalue_list_unadj_worsesurvival
}

## ===============================
## (Optional) Save results
## ===============================
tissue_type_df <- do.call(rbind, tissue_type_list)
tissue_type_df$type <- ifelse(tissue_type_df$tumor_descriptor == "Not Applicable", "Normal", "Tumor")
write.csv(tissue_type_df, file = paste0(file_path_main, "TCGA_tissue_type_annotation_", MI_type, ".csv"), row.names = TRUE)

nsample_vec_df <- data.frame(
  Project = names(nsample_vec),
  N_Samples = unlist(nsample_vec)
)

MI_decomposition_all <- Reduce(rbind, MI_decomposition_list)
cancertype_label_all <- unlist(lapply(names(MI_decomposition_list), function(x) {
  rep(x, nrow(MI_decomposition_list[[x]]))
}))

save(MI_decomposition_list, file = paste0(file_path_main, "MI_decomposition_list_", MI_type, ".Rdata"))
save(clinical_data_list, file = paste0(file_path_main, "clinical_data_list_", MI_type, ".Rdata"))

rownames(HR_mat) <- MI_unique
colnames(HR_mat) <- TCGA_project_list
rownames(pvalue_mat) <- MI_unique
colnames(pvalue_mat) <- TCGA_project_list
rownames(pvalue_mat_bettersurvival) <- MI_unique
colnames(pvalue_mat_bettersurvival) <- TCGA_project_list
rownames(pvalue_mat_worsesurvival) <- MI_unique
colnames(pvalue_mat_worsesurvival) <- TCGA_project_list

rownames(HR_mat_unadj) <- MI_unique
colnames(HR_mat_unadj) <- TCGA_project_list
rownames(pvalue_mat_unadj) <- MI_unique
colnames(pvalue_mat_unadj) <- TCGA_project_list
rownames(pvalue_mat_unadj_bettersurvival) <- MI_unique
colnames(pvalue_mat_unadj_bettersurvival) <- TCGA_project_list
rownames(pvalue_mat_unadj_worsesurvival) <- MI_unique
colnames(pvalue_mat_unadj_worsesurvival) <- TCGA_project_list

write.csv(HR_mat, file = paste0(file_path_main, "TCGA_MIs_HR_matrix_", MI_type, ".csv"))
write.csv(pvalue_mat, file = paste0(file_path_main, "TCGA_MIs_pvalue_matrix_", MI_type, ".csv"))
write.csv(pvalue_mat_bettersurvival, file = paste0(file_path_main, "TCGA_MIs_pvalue_bettersurvival_matrix_", MI_type, ".csv"))
write.csv(pvalue_mat_worsesurvival, file = paste0(file_path_main, "TCGA_MIs_pvalue_worsesurvival_matrix_", MI_type, ".csv"))

write.csv(HR_mat_unadj, file = paste0(file_path_main, "TCGA_MIs_HR_matrix_unadjusted_", MI_type, ".csv"))
write.csv(pvalue_mat_unadj, file = paste0(file_path_main, "TCGA_MIs_pvalue_matrix_unadjusted_", MI_type, ".csv"))
write.csv(pvalue_mat_unadj_bettersurvival, file = paste0(file_path_main, "TCGA_MIs_pvalue_bettersurvival_matrix_unadjusted_", MI_type, ".csv"))
write.csv(pvalue_mat_unadj_worsesurvival, file = paste0(file_path_main, "TCGA_MIs_pvalue_worsesurvival_matrix_unadjusted_", MI_type, ".csv"))

write.csv(nsample_vec_df, file = paste0(file_path_main, "TCGA_nsample_vec_", MI_type, ".csv"), row.names = FALSE)
# ##Compare the HR between Markergene and MIdecomposition
# HR_mat_markergene <- read.csv(paste0(file_path_main, "TCGA_MIs_HR_matrix_Markergene.csv"), row.names=1)
# HR_mat_MIdecomposition <- read.csv(paste0(file_path_main, "TCGA_MIs_HR_matrix_MIdecomposition.csv"), row.names=1)
# pvalue_mat_markergene <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_matrix_Markergene.csv"), row.names=1)
# pvalue_mat_MIdecomposition <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_matrix_MIdecomposition.csv"), row.names=1)
# 
# 
# ##Scatter plot between as.vector(as.matrix(HR_mat_markergene)) and as.vector(as.matrix(HR_mat_MIdecomposition))
# library(ggplot2)
# df_HR_compare <- data.frame(
#   HR_markergene = as.vector(as.matrix(HR_mat_markergene)),
#   HR_MIdecomposition = as.vector(as.matrix(HR_mat_MIdecomposition))
# )
# p1<- ggplot(df_HR_compare, aes(x=HR_markergene, y=HR_MIdecomposition)) +
#   geom_point() +
#   geom_smooth(method='lm', formula= y~x, color='blue') +
#   labs(title='Comparison of HR between Markergene and MIdecomposition',
#        x='HR (Markergene)',
#        y='HR (MIdecomposition)') +
#   theme_minimal()+
#   xlim(0.6, 1.5)+
#   ylim(0.6, 1.5)
# ##Save the plot
# ggsave(paste0(file_path_main, "HR_comparison_Markergene_vs_MIdecomposition.png"), plot = p1, width = 6, height = 5, dpi = 300)
# 
# 
# corr_value <- cor(as.vector(as.matrix(HR_mat_markergene)), as.vector(as.matrix(HR_mat_MIdecomposition)), method = "pearson", use = "complete.obs")
# corr_value
# corr_value <- cor(as.vector(as.matrix(HR_mat_markergene)), as.vector(as.matrix(HR_mat_MIdecomposition)), method = "spearman", use = "complete.obs")
# corr_value
# 
# ##Scatter plot between as.vector(as.matrix(-log10(pvalue_mat_markergene))) and as.vector(as.matrix(-log10(pvalue_mat_MIdecomposition)))
# df_pvalue_compare <- data.frame(
#   neglog10_pvalue_markergene = -log10(as.vector(as.matrix(pvalue_mat_markergene))),
#   neglog10_pvalue_MIdecomposition = -log10(as.vector(as.matrix(pvalue_mat_MIdecomposition)))
# )
# df_pvalue_compare$neglog10_pvalue_markergene <- df_pvalue_compare$neglog10_pvalue_markergene * ifelse(as.vector(as.matrix(HR_mat_markergene)) >1,1,-1)
# df_pvalue_compare$neglog10_pvalue_MIdecomposition <- df_pvalue_compare$neglog10_pvalue_MIdecomposition * ifelse(as.vector(as.matrix(HR_mat_MIdecomposition)) >1,1,-1)
# p2<- ggplot(df_pvalue_compare, aes(x=neglog10_pvalue_markergene, y=neglog10_pvalue_MIdecomposition)) +
#   geom_point() +
#   geom_smooth(method='lm', formula= y~x, color='blue') +
#   labs(title='Comparison of -log10(pvalue) between Markergene and MIdecomposition',
#        x='-log10(pvalue) (Markergene)',
#        y='-log10(pvalue) (MIdecomposition)') +
#   theme_minimal()+
#   xlim(-4,4)+
#   ylim(-4,4)
# ##Save the plot
# ggsave(paste0(file_path_main, "pvalue_comparison_Markergene_vs_MIdecomposition.png"), plot = p2, width = 6, height = 5, dpi = 300)
# 
# corr_value <- cor(as.vector(df_pvalue_compare$neglog10_pvalue_markergene), as.vector(df_pvalue_compare$neglog10_pvalue_MIdecomposition), method = "spearman", use = "complete.obs")
# corr_value

##Load the HR and pvalue matrix
HR_mat <- read.csv(paste0(file_path_main, "TCGA_MIs_HR_matrix_",MI_type,".csv"), row.names=1)
pvalue_mat <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_matrix_",MI_type,".csv"), row.names=1)
pvalue_mat_bettersurvival <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_bettersurvival_matrix_",MI_type,".csv"), row.names=1)
pvalue_mat_worsesurvival <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_worsesurvival_matrix_",MI_type,".csv"), row.names=1)
# ##load the unadjusted HR and pvalue matrix
# HR_mat <- read.csv(paste0(file_path_main, "TCGA_MIs_HR_matrix_unadjusted_",MI_type,".csv"), row.names=1)
# pvalue_mat <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_matrix_unadjusted_",MI_type,".csv"), row.names=1)
# pvalue_mat_bettersurvival <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_bettersurvival_matrix_unadjusted_",MI_type,".csv"), row.names=1)
# pvalue_mat_worsesurvival <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_worsesurvival_matrix_unadjusted_",MI_type,".csv"), row.names=1)

# HR_mat <- read.csv(paste0(file_path_main, "TCGA_MIs_HR_matrix_unadjusted_",MI_type,".csv"), row.names=1)
# pvalue_mat <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_matrix_unadjusted_",MI_type,".csv"), row.names=1)
# pvalue_mat_bettersurvival <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_bettersurvival_matrix_unadjusted_",MI_type,".csv"), row.names=1)
# pvalue_mat_worsesurvival <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_worsesurvival_matrix_",MI_type,".csv"), row.names=1)

nsample_vec_df <- read.csv(paste0(file_path_main, "TCGA_nsample_vec_",MI_type,".csv"))
# HR_mat <- read.csv(paste0(file_path_main, "TCGA_MIs_HR_matrix_","Markergene",".csv"), row.names=1)
# pvalue_mat <- read.csv(paste0(file_path_main, "TCGA_MIs_pvalue_matrix_","Markergene",".csv"), row.names=1)

HR_mat_log2 <- log2(HR_mat)

# ##Load the Avg_MI_cellclass_pair_heatmap_values to get the MI order
# Avg_MI_cellclass_pair_heatmap_values <- read.csv(paste0(file_path_main, "Avg_MI_cellclass_pair_heatmap_values.csv"), row.names=1)
# MI_order <- rownames(Avg_MI_cellclass_pair_heatmap_values)
# ##Test
# Avg_MI_cellclass_pair_heatmap_values_choose<-Avg_MI_cellclass_pair_heatmap_values['MI-5',]
# ##Sort rge Avg_MI_cellclass_pair_heatmap_values_choose
# Avg_MI_cellclass_pair_heatmap_values_choose <- Avg_MI_cellclass_pair_heatmap_values_choose[ ,order(-as.numeric(Avg_MI_cellclass_pair_heatmap_values_choose))]
# Avg_MI_cellclass_pair_heatmap_values_choose <- t(Avg_MI_cellclass_pair_heatmap_values_choose)
# head(Avg_MI_cellclass_pair_heatmap_values_choose,20)

MI_order <- c("MI-2","MI-10","MI-8","MI-9","MI-1","MI-7",
              "MI-6","MI-3","MI-5","MI-4","MI-11")
##Replace the "MI-" to "MI"
MI_order <- gsub("-","", MI_order)

HR_mat_log2 <- HR_mat_log2[MI_order, ]
pvalue_mat <- pvalue_mat[MI_order, ]
pvalue_mat_bettersurvival <- pvalue_mat_bettersurvival[MI_order, ]
pvalue_mat_worsesurvival <- pvalue_mat_worsesurvival[MI_order, ]
negpvalue_mat <- -log10(pvalue_mat)
negpvalue_mat_bettersurvival <- -log10(pvalue_mat_bettersurvival)
negpvalue_mat_worsesurvival <- -log10(pvalue_mat_worsesurvival)
negpvalue_mat_merge <-ifelse(as.matrix(negpvalue_mat_worsesurvival) > as.matrix(negpvalue_mat_bettersurvival),as.matrix(negpvalue_mat_worsesurvival), (-1) * as.matrix(negpvalue_mat_bettersurvival))
HR_mat <- HR_mat[MI_order, ]
##Show the heatmap of negpvalue_mat_merge,color from blue to white to red
library(pheatmap)

## 可选：先把数值截断到 [-2, 2]
negpvalue_mat_plot <- negpvalue_mat_merge
# negpvalue_mat_plot[negpvalue_mat_plot >  2] <-  2
# negpvalue_mat_plot[negpvalue_mat_plot < -2] <- -2

thr <- -log10(0.05)  # 1.30103

# 非均匀 breaks：中间更密
my_breaks <- c(
  seq(-2,   -thr, length.out = 60),
  seq(-thr,  thr, length.out = 120),  # ⭐ 中间更密
  seq( thr,   2, length.out = 60)
)
my_breaks <- unique(my_breaks)

# 对应颜色（中间白色 plateau 更宽）
my_colors <- colorRampPalette(
  c("#4D95C2", "#87B0D3", "white", "#F7A199", "#E9242D")
)(length(my_breaks) - 1)

pheatmap(
  negpvalue_mat_plot,
  color = my_colors,
  breaks = my_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  fontsize = 16,
  fontsize_row  = 16,
  fontsize_col  = 16,
  filename = paste0(
    paste0(file_path_main, "TCGA_MIs_neglog10pvalue_heatmap_"),
    MI_type, ".pdf"
  ),
  width = 6,
  height = 8
)



###########
library(pheatmap)

## 可选：先把数值截断到 [-2, 2]
negpvalue_mat_plot <- negpvalue_mat_merge
# negpvalue_mat_plot[negpvalue_mat_plot >  2] <-  2
# negpvalue_mat_plot[negpvalue_mat_plot < -2] <- -2

thr <- -log10(0.05)  # 1.30103

# 非均匀 breaks：中间更密
my_breaks <- c(
  seq(-2,   -thr, length.out = 60),
  seq(-thr,  thr, length.out = 120),
  seq( thr,   2, length.out = 60)
)
my_breaks <- unique(my_breaks)

# 对应颜色
my_colors <- colorRampPalette(
  c("#4D95C2", "#87B0D3", "white", "#F7A199", "#E9242D")
)(length(my_breaks) - 1)

## 标记绝对值大于 thr 的格子
sig_mask <- abs(negpvalue_mat_plot) > thr

## 用黄色方块做高亮标记
display_mat <- matrix(
  ifelse(sig_mask, "■", ""),
  nrow = nrow(negpvalue_mat_plot),
  ncol = ncol(negpvalue_mat_plot)
)
rownames(display_mat) <- rownames(negpvalue_mat_plot)
colnames(display_mat) <- colnames(negpvalue_mat_plot)

pheatmap(
  negpvalue_mat_plot,
  color = my_colors,
  breaks = my_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  fontsize = 16,
  fontsize_row = 16,
  fontsize_col = 16,
  
  display_numbers = display_mat,
  number_color = "yellow",
  fontsize_number = 10,
  
  filename = paste0(
    file_path_main,
    "TCGA_MIs_neglog10pvalue_heatmap_V2_",
    MI_type,
    ".pdf"
  ),
  width = 6,
  height = 8
)


library(pheatmap)

############################
## 1. 准备显著性矩阵
############################
negpvalue_mat_plot <- negpvalue_mat_merge
# negpvalue_mat_plot[negpvalue_mat_plot >  2] <-  2
# negpvalue_mat_plot[negpvalue_mat_plot < -2] <- -2

thr <- -log10(0.05)  # 1.30103

############################
## 2. 准备 HR heatmap 矩阵
############################
HR_mat_plot <- HR_mat

## 可选：把 HR 截断到 [0, 2]
HR_mat_plot[HR_mat_plot > 2] <- 2
HR_mat_plot[HR_mat_plot < 0] <- 0

############################
## 3. 定义以 1 为中心的 breaks
############################
my_breaks <- c(
  seq(0, 1, length.out = 100),
  seq(1, 2, length.out = 100)
)
my_breaks <- unique(my_breaks)

############################
## 4. 定义颜色：中心点 1 对应白色
############################
my_colors <- colorRampPalette(
  c("#4D95C2", "white", "#E9242D")
)(length(my_breaks) - 1)

############################
## 5. 显著性 mask
############################
sig_mask <- abs(negpvalue_mat_plot) > thr

## 检查维度是否一致
if (!all(dim(HR_mat_plot) == dim(sig_mask))) {
  stop("Dimensions of HR_mat and negpvalue_mat_plot must match.")
}

if (!identical(rownames(HR_mat_plot), rownames(negpvalue_mat_plot)) ||
    !identical(colnames(HR_mat_plot), colnames(negpvalue_mat_plot))) {
  stop("Row/column names of HR_mat and negpvalue_mat_plot must match.")
}

############################
## 6. 构造黄色 mask 标记
############################
display_mat <- matrix(
  ifelse(sig_mask, "■", ""),
  nrow = nrow(HR_mat_plot),
  ncol = ncol(HR_mat_plot)
)
rownames(display_mat) <- rownames(HR_mat_plot)
colnames(display_mat) <- colnames(HR_mat_plot)

############################
## 7. 绘图
############################
pheatmap(
  HR_mat_plot,
  color = my_colors,
  breaks = my_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  fontsize = 16,
  fontsize_row = 16,
  fontsize_col = 16,
  
  display_numbers = display_mat,
  number_color = "yellow",
  fontsize_number = 10,
  
  filename = paste0(
    file_path_main,
    "TCGA_MIs_HR_heatmap_with_sigmask_",
    MI_type,
    ".pdf"
  ),
  width = 6,
  height = 8
)

# ## 可选：先把数值截断到 [-2, 2]
# negpvalue_mat_plot <- negpvalue_mat_merge
# # negpvalue_mat_plot[negpvalue_mat_plot >  2] <-  2
# # negpvalue_mat_plot[negpvalue_mat_plot < -2] <- -2
# 
# thr <- -log10(0.05)  # 1.30103
# 
# ## breaks
# breaks_low  <- seq(-2,  -thr, length.out = 60)
# breaks_mid  <- seq(-thr, thr, length.out = 40)
# breaks_high <- seq( thr,  2, length.out = 60)
# 
# my_breaks <- unique(c(breaks_low, breaks_mid, breaks_high))
# 
# ## colors
# cols_low  <- colorRampPalette(c("#4D95C2", "#87B0D3"))(length(breaks_low)-1)
# cols_mid  <- rep("#D3D3D3", length(breaks_mid)-1)   # ⭐ 灰色 plateau
# cols_high <- colorRampPalette(c("#F7A199", "#E9242D"))(length(breaks_high)-1)
# 
# my_colors <- c(cols_low, cols_mid, cols_high)
# 
# pheatmap(
#   negpvalue_mat_plot,
#   color = my_colors,
#   breaks = my_breaks,
#   cluster_rows = FALSE,
#   cluster_cols = FALSE,
#   fontsize = 16,
#   fontsize_row  = 16,
#   fontsize_col  = 16,
#   filename = paste0(
#     file_path_main,
#     "TCGA_MIs_neglog10pvalue_heatmap_",
#     MI_type,
#     ".pdf"
#   ),
#   width = 6,
#   height = 8
# )
# 

# negpvalue_mat_plot_sd <- apply(negpvalue_mat_plot,MARGIN = 1, sd)
negpvalue_mat_plot_sd <- apply(abs(negpvalue_mat_plot),MARGIN = 1, sd)

library(ggplot2)

df <- data.frame(
  MI = factor(names(negpvalue_mat_plot_sd),
              levels = names(negpvalue_mat_plot_sd)),
  SD = as.numeric(negpvalue_mat_plot_sd)
)

# 分组：前 4 vs 其余
df$group <- c(rep("Top4", 4), rep("Others", nrow(df) - 4))

mean_first4 <- mean(df$SD[1:4], na.rm = TRUE)
mean_rest   <- mean(df$SD[-(1:4)], na.rm = TRUE)

p1 <- ggplot(df, aes(x = MI, y = SD, fill = group)) +
  geom_col(width = 0.7) +
  geom_hline(
    yintercept = mean_first4,
    linetype = "dashed",
    linewidth = 1,
    color = "#5A9CB5"
  ) +
  geom_hline(
    yintercept = mean_rest,
    linetype = "dashed",
    linewidth = 1,
    color = "#FA6868"
  ) +
  scale_fill_manual(
    values = c(
      "Top4" = "#5A9CB5",
      "Others" = "#FA6868"
    )
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_blank()
  ) +
  labs(
    x = "",
    y = "SD of -log10(p-value)",
    title = "Prognostic Variability Across Meta-Interactions"
  )

ggsave(
  filename = paste0(
    file_path_main, "TCGA_MIs_neglog10pvalue_SD_histogram_",
    MI_type, ".pdf"
  ),
  plot = p1,
  width = 6,
  height = 4,
  dpi = 300
)



library(ggplot2)

df <- data.frame(
  MI = factor(names(negpvalue_mat_plot_sd),
              levels = names(negpvalue_mat_plot_sd)),
  SD = as.numeric(negpvalue_mat_plot_sd)
)
df$group <- c(rep("Top4", 4), rep("Others", nrow(df) - 4))

# Wilcoxon rank-sum test
wt <- wilcox.test(SD ~ group, data = df, exact = FALSE)

p_text <- paste0("Wilcoxon rank-sum p = ", formatC(wt$p.value, format = "e", digits = 2))

p_box <- ggplot(df, aes(x = group, y = SD, fill = group)) +
  geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.8) +
  geom_jitter(width = 0.15, size = 2, alpha = 0.9) +
  scale_fill_manual(values = c("Top4" = "#5A9CB5", "Others" = "#FA6868")) +
  theme_classic(base_size = 14) +
  theme(legend.position = "none") +
  labs(
    x = "",
    y = "SD of -log10(p-value)",
    title = "Prognostic Variability of Meta-Interactions"
  ) +
  annotate(
    "text",
    x = 1.5,
    y = max(df$SD, na.rm = TRUE) * 1.05,
    label = p_text,
    size = 4
  ) +
  coord_cartesian(ylim = c(min(df$SD, na.rm = TRUE), max(df$SD, na.rm = TRUE) * 1.12))

p_box
ggsave(
  filename = paste0(
    file_path_main, "TCGA_MIs_neglog10pvalue_SD_boxplot_",
    MI_type, ".pdf"
  ),
  plot = p_box,
  width = 4,
  height = 5,
  dpi = 300
)



##Combine p value

stouffer_combine_onesided_mat <- function(p_mat, nsample_vec_df) {
  p_mat <- as.matrix(p_mat)
  mode(p_mat) <- "numeric"
  
  w_all <- as.numeric(nsample_vec_df$N_Samples)
  # w_all <- sqrt(w_all)  # Stouffer's method typically uses sqrt of sample size as weights
  
  # 如果权重长度不等于列数，直接报错，避免静默错位
  if (length(w_all) != ncol(p_mat)) {
    stop("Length of nsample_vec_df$N_Samples must equal ncol(p_mat).")
  }
  
  stouffer_row <- function(p_vec) {
    # 过滤 NA / 非(0,1) 的值
    valid <- is.finite(p_vec) & p_vec > 0 & p_vec < 1
    k <- sum(valid)
    if (k == 0) return(c(z = NA_real_, p = NA_real_))
    
    p_use <- p_vec[valid]
    w_use <- w_all[valid]
    
    # 单边 p -> Z（越小越显著 => Z 越大）
    z_vals <- qnorm(1 - p_use)
    
    # 加权 Stouffer：标准形式
    z_comb <- sum(w_use * z_vals) / sqrt(sum(w_use^2))
    
    # 合并后的单边 p
    p_comb <- 1 - pnorm(z_comb)
    
    c(z = z_comb, p = p_comb)
  }
  
  res <- t(apply(p_mat, 1, stouffer_row))
  res <- as.data.frame(res)
  if (!is.null(rownames(p_mat))) rownames(res) <- rownames(p_mat)
  res
}



## ============================
## 对 better survival 的 p 矩阵
## ============================
stouffer_better <- stouffer_combine_onesided_mat(pvalue_mat_bettersurvival,nsample_vec_df)

## ============================
## 对 worse survival 的 p 矩阵
## ============================
stouffer_worse  <- stouffer_combine_onesided_mat(pvalue_mat_worsesurvival,nsample_vec_df)

## 查看结果
head(stouffer_better)
head(stouffer_worse)

## 如果想把两种方向放在一起：
stouffer_summary <- data.frame(
  MI = rownames(stouffer_better),
  p_better = stouffer_better$p,
  p_worse  = stouffer_worse$p
)
stouffer_summary$neglog10p_better <- -log10(stouffer_summary$p_better)
stouffer_summary$neglog10p_worse  <- -log10(stouffer_summary$p_worse)

stouffer_summary$overall_sign <- ifelse(
  stouffer_summary$neglog10p_better > stouffer_summary$neglog10p_worse,
  "better survival",
  "worse survival"
)
stouffer_summary$p_final <- ifelse(
  stouffer_summary$overall_sign == "better survival",
  stouffer_summary$p_better,
  stouffer_summary$p_worse
)
stouffer_summary$neglog10p_final <- -log10(stouffer_summary$p_final)
stouffer_summary$neglog10p_final_signed <- ifelse(
  stouffer_summary$overall_sign == "worse survival",
  stouffer_summary$neglog10p_final,
  -stouffer_summary$neglog10p_final
)

stouffer_summary
##Show the barplot of neglog10p_final_signed
## 数值 & 颜色
## 数值 & 颜色 ---------------------------------------------------------
vals <- stouffer_summary$neglog10p_final_signed
vals_abs <- abs(vals)
names(vals) <- stouffer_summary$MI

cols <- ifelse(vals >= 0, "#E9242D", "#4D95C2")

## 阈值：p = 0.05 ------------------------------------------------------
thr_pos <- -log10(0.05)  # 正侧阈值
# thr_neg <-  log10(0.05)  # 负侧阈值（约 -1.301）

## 对称 x 轴范围 --------------------------------------------------------
max_abs <- max(abs(vals), abs(thr_pos), na.rm = TRUE)

## 反转顺序（y 轴从上到下按原始顺序） ----------------------------------
vals_rev  <- rev(vals)
vals_abs_rev  <- rev(vals_abs)
names_rev <- rev(names(vals))
cols_rev  <- rev(cols)

## 保存到文件 -----------------------------------------------------------
pdf(
  file = paste0(
    file_path_main, "TCGA_MIs_Stouffer_barplot_",
    MI_type, ".pdf"
  ),
  width = 3,
  height = 6
)

bp <- barplot(
  # vals_rev,
  vals_abs_rev,
  names.arg = names_rev,
  horiz = TRUE,                    # 横向条
  # xlim  = c(-max_abs, max_abs),
  xlim  = c(0, max_abs),
  las   = 1,                       # y 轴文字水平
  col   = cols_rev,
  border = NA,
  axes   = FALSE,                  # 先关掉默认坐标轴
  xlab   = "",
  ylab   = ""
)

## x 轴刻度线 -----------------------------------------------------------
xticks <- pretty(c(-max_abs, max_abs))   # 自动生成一组好看的刻度
axis(1, at = xticks, labels = xticks)    # 只画 x 轴刻度和刻度线

## 阈值线：±log10(0.05) & 0 --------------------------------------------
abline(v = thr_pos, lty = 2)
# abline(v = thr_neg, lty = 2)
abline(v = 0,       lty = 1)

dev.off()


##rotation
vals <- stouffer_summary$neglog10p_final_signed
vals_abs <- abs(vals)
names(vals) <- stouffer_summary$MI

# cols <- ifelse(vals >= 0, "#E9242D", "#4D95C2")
col_pos_strong <- "#E9242D"  # 深红
col_pos_weak   <- "#F4A3A8"  # 浅红
col_neg_strong <- "#4D95C2"  # 深蓝
col_neg_weak   <- "#A9CBE6"  # 浅蓝
## 颜色：方向（正/负）+ 显著性（深/浅） -------------------------------
cols <- ifelse(
  vals >= 0 & abs(vals) >= thr_pos, col_pos_strong,
  ifelse(
    vals >= 0 & abs(vals) <  thr_pos, col_pos_weak,
    ifelse(
      vals <  0 & abs(vals) >= thr_pos, col_neg_strong,
      col_neg_weak
    )
  )
)


## 阈值：p = 0.05 ------------------------------------------------------
thr_pos <- -log10(0.05)

## 对称 y 轴范围 --------------------------------------------------------
max_abs <- max(abs(vals), abs(thr_pos), na.rm = TRUE)

## 保持原顺序（x 轴从左到右） -----------------------------------------
vals_abs_use <- vals_abs
names_use <- names(vals)
cols_use <- cols

## 保存到文件 ----------------------------------------------------------
pdf(
  file = paste0(
    file_path_main, "TCGA_MIs_Stouffer_barplot_",
    MI_type, "_vertical.pdf"
  ),
  width = 6,
  height = 3.2
)
par(xaxs = "i")

bp <- barplot(
  vals_abs_use,
  names.arg = names_use,
  ylim  = c(0, max_abs * 1.6),
  las   = 2,              # x 轴文字垂直（MI 多时必须）
  col   = cols_use,
  border = "black",
  axes   = FALSE,
  xlab   = "",
  ylab   = ""
)

## y 轴刻度 ------------------------------------------------------------
yticks <- pretty(c(0, max_abs),n = 3)
axis(2, at = yticks, labels = yticks, las = 1)

## 阈值线：log10(0.05) & 0 ---------------------------------------------
abline(h = thr_pos, lty = 2)
abline(h = 0,       lty = 1)

box(col = "black", lwd = 1)

dev.off()



#########



rowSums(pvalue_mat_bettersurvival < 0.05)
rowSums(pvalue_mat_worsesurvival < 0.05)
##TCGA_cancertype_order
TCGA_cancertype_choose <- c("TCGA.BRCA",
                            "TCGA.COAD",
                            "TCGA.LIHC",
                            "TCGA.LUAD","TCGA.LUSC",
                            "TCGA.SKCM",
                            "TCGA.OV",
                            "TCGA.PRAD",
                            "TCGA.UCEC")
TCGA_cancertype_other <- setdiff(colnames(HR_mat_log2), TCGA_cancertype_choose)
TCGA_cancertype_order <- c(TCGA_cancertype_choose, TCGA_cancertype_other)

HR_mat_log2 <- HR_mat_log2[, TCGA_cancertype_order]
##Fill the NA value with 0
HR_mat_log2[is.na(HR_mat_log2)] <- 0
pvalue_mat <- pvalue_mat[, TCGA_cancertype_order]
negpvalue_mat <- negpvalue_mat[, TCGA_cancertype_order]

filter_mat <-(HR_mat_log2 > log2(1)) & (pvalue_mat < 0.10)
filter_mat <-(HR_mat_log2 < log2(1)) & (pvalue_mat < 0.10)
# filter_mat <-(HR_mat_log2 > 0)
filter_mat_rowsum <- rowSums(filter_mat)
filter_mat_rowsum

pvalue_mat_bettersurvival <- 
  
  sd_vec <- apply(HR_mat_log2, MARGIN = 1,sd)
##Show the barplot of sd_vec
barplot(sd_vec, main="SD of log2(HR) of MIs across cancers", ylab="SD of log2(HR)", xlab="MIs (ordered by SD)", las=2, cex.names=0.7)
library(reshape2)
library(dplyr)
library(ggplot2)

# HR_mat_log2 <- as.data.frame(
#   pmax(pmin(HR_mat_log2, 2), -2)
# )
HR_mat_log2 <- as.data.frame(
  pmax(pmin(HR_mat_log2, log2(1.5)), -log2(1.5))
)

negpvalue_mat <- as.data.frame(
  pmin(HR_mat_log2, 6)
)



# HR matrix → long
df_hr <- melt(as.matrix(HR_mat_log2))
colnames(df_hr) <- c("MI", "Cancer", "HR_log2")

# neg p-value matrix → long
df_p <- melt(as.matrix(negpvalue_mat))
colnames(df_p) <- c("MI", "Cancer", "neglog10p")

# 合并
df_plot <- df_hr %>%
  left_join(df_p, by = c("MI", "Cancer"))



df_plot$MI <- factor(
  df_plot$MI,
  levels = rev(rownames(HR_mat_log2))   # 让 MI1 在上面可去掉 rev
)

df_plot$Cancer <- factor(
  df_plot$Cancer,
  levels = colnames(HR_mat_log2)
)


p <- ggplot(df_plot, aes(x = Cancer, y = MI)) +
  geom_vline(xintercept = seq_along(levels(df_plot$Cancer)),
             color = "grey90", linewidth = 0.3) +
  geom_hline(yintercept = seq_along(levels(df_plot$MI)),
             color = "grey90", linewidth = 0.3)+
  geom_point(
    aes(
      color = HR_log2,
      size  = neglog10p
    ),
    alpha = 0.9
  ) +
  scale_color_gradient2(
    low = "#4575b4",     # 蓝
    mid = "white",       # 0 → 白
    high = "#d73027",    # 红
    midpoint = 0,
    name = "log2(HR)"
  ) +
  scale_size_continuous(
    range = c(1, 6),
    name = "-log10(p)"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.title = element_blank(),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10)
  )

p
# ggsave("D:/CellFlowMap/Pancancer/Results/V1/SpiderNet_Result_Mode_cell_class_selfsetdim12/TCGA_MIs_HR_heatmap.pdf",
ggsave(paste0(file_path_main, "TCGA_MIs_HR_heatmap_",MI_type,".pdf"),
       plot = p,
       width = 10,
       height = 8)





##Get the intersected feature in clinical_feature_list
load(file=paste0(file_path_main, "MI_decomposition_list_",MI_type,".Rdata"))
load(file=paste0(file_path_main, "clinical_data_list_",MI_type,".Rdata"))

Avg_MI_cellclass_pair_heatmap_values <- read.csv( paste0(file_path_main, "Avg_MI_cellclass_pair_heatmap_values.csv"), row.names=1)
MI_order <- rownames(Avg_MI_cellclass_pair_heatmap_values)


clinical_feature_list <- list()
for (i in 1:length(clinical_data_list)){
  clinical_data_cur <- clinical_data_list[[i]]
  clinical_feature_list[[i]] <- colnames(clinical_data_cur)
}
clinical_feature_all <- Reduce(union, clinical_feature_list)

tumor_severity_features <- c(
  "ajcc_pathologic_stage",
  "ajcc_pathologic_t",
  "ajcc_pathologic_n",
  "ajcc_pathologic_m",
  "figo_stage",
  "tumor_grade",
  "paper_pathologic_stage",
  "paper_Tumor.stage",
  "paper_Tumor_Grade"
)

immune_context_features <- c(
  "paper_MSI_status",
  "paper_msi_status_7_marker_call",
  "paper_hypermutated",
  "paper_TOTAL.MUTATIONS",
  "paper_nonsilent_mutrate"
)

genome_instability_features <- c(
  "paper_CNV Clusters",
  "paper_cna_cluster_k4",
  "paper_Fraction_genome_altered",
  "paper_Absolute_Ploidy",
  "paper_ABSOLUTE_purity"
)

molecular_subtype_features <- c(
  "paper_BRCA_Subtype_PAM50",     # breast
  "paper_IntegrativeCluster",
  "paper_mRNA Clusters",
  "paper_mRNA_cluster",
  "paper_Protein Clusters",
  "paper_RPPA_cluster",
  "paper_methylation_cluster",
  "paper_PARADIGM Clusters"
)

tme_relevant_features <- c(
  "paper_LYMPHOCYTE.DENSITY",
  "paper_LYMPHOCYTE.SCORE",
  "paper_PURITY..ABSOLUTE.",
  "paper_Tumour.content........nuceli.that.are.tumour_cells......0.100.."
)


clinical_feature_useful <- c(
  tumor_severity_features,
  immune_context_features,
  genome_instability_features,
  molecular_subtype_features,
  tme_relevant_features
)

# [1] "ajcc_pathologic_stage"                                                 "ajcc_pathologic_t"                                                    
#  [3] "ajcc_pathologic_n"                                                     "ajcc_pathologic_m"

## ===============================
## AJCC ordered mappings
## ===============================

ajcc_stage_levels <- c(
  "Stage 0",
  "Stage I", "Stage IA", "Stage IB", "Stage IC",
  "Stage II", "Stage IIA", "Stage IIB", "Stage IIC",
  "Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC",
  "Stage IV", "Stage IVA", "Stage IVB"
)
ajcc_stage_numeric_map <- setNames(seq_along(ajcc_stage_levels), ajcc_stage_levels)

ajcc_t_levels <- c(
  "Tis","T0",
  "T1","T1a","T1b","T1b1","T1c",
  "T2","T2a","T2b","T2c",
  "T3","T3a","T3b",
  "T4","T4a","T4b","T4d"
)
ajcc_t_numeric_map <- setNames(seq_along(ajcc_t_levels), ajcc_t_levels)

ajcc_n_levels <- c(
  "N0",
  "N0 (i-)","N0 (i+)","N0 (mol+)",
  "N1","N1mi","N1a","N1b","N1c",
  "N2","N2a","N2b","N2c",
  "N3","N3a","N3b"
)
ajcc_n_numeric_map <- setNames(seq_along(ajcc_n_levels), ajcc_n_levels)

ajcc_m_levels <- c(
  "M0",
  "cM0 (i+)",
  "M1","M1a","M1b","M1c"
)
ajcc_m_numeric_map <- setNames(seq_along(ajcc_m_levels), ajcc_m_levels)

ajcc_ordered_features <- c(
  "ajcc_pathologic_stage",
  "ajcc_pathologic_t",
  "ajcc_pathologic_n",
  "ajcc_pathologic_m"
)

## ===============================
## Result containers
## ===============================

pvalue_res_mat <- matrix(
  NA,
  nrow = length(clinical_feature_useful),
  ncol = ncol(MI_decomposition_list[[1]]),
  dimnames = list(
    clinical_feature_useful,
    colnames(MI_decomposition_list[[1]])
  )
)

statistic_res_mat <- matrix(
  NA,
  nrow = length(clinical_feature_useful),
  ncol = ncol(MI_decomposition_list[[1]]),
  dimnames = list(
    clinical_feature_useful,
    colnames(MI_decomposition_list[[1]])
  )
)

feature_type_vec <- c()

## ===============================
## Main loop
## ===============================

for (clinical_feature_cur in clinical_feature_useful) {
  
  message("Processing: ", clinical_feature_cur)
  
  ## find datasets containing this feature
  clinical_feature_index <- which(
    sapply(clinical_feature_list, function(x)
      clinical_feature_cur %in% x)
  )
  
  if (length(clinical_feature_index) == 0) {
    feature_type_vec <- c(feature_type_vec, NA)
    next
  }
  
  clinical_feature_cur_vec <- c()
  MI_decomposition_list_cur <- list()
  
  for (k in clinical_feature_index) {
    name_cur <- names(clinical_data_list)[k]
    clinical_data_cur <- clinical_data_list[[name_cur]]
    MI_decomposition_cur <- MI_decomposition_list[[name_cur]]
    
    clinical_data_cur <- clinical_data_cur[rownames(MI_decomposition_cur), , drop = FALSE]
    clinical_feature_cur_vec <- c(
      clinical_feature_cur_vec,
      clinical_data_cur[, clinical_feature_cur]
    )
    
    MI_decomposition_list_cur[[name_cur]] <- MI_decomposition_cur
  }
  
  MI_decomposition_list_cur_all <- Reduce(rbind, MI_decomposition_list_cur)
  
  ## determine feature type
  if (clinical_feature_cur %in% ajcc_ordered_features) {
    feature_type <- "ordered_continuous"
  } else if (is.numeric(clinical_feature_cur_vec)) {
    feature_type <- "continuous"
  } else {
    feature_type <- "categorical"
  }
  
  feature_type_vec <- c(feature_type_vec, feature_type)
  
  pvalue_res_vec <- c()
  statistic_res_vec <- c()
  
  for (MI_index in seq_len(ncol(MI_decomposition_list_cur_all))) {
    
    MI_value <- MI_decomposition_list_cur_all[, MI_index]
    MI_median <- median(MI_value, na.rm = TRUE)
    # threshold_score <- MI_median
    # threshold_score <- quantile(MI_value,0.75, na.rm = TRUE)
    threshold_score <- quantile(MI_value,0.3, na.rm = TRUE)
    # threshold_score <- 0.5
    MI_group <- ifelse(MI_value >= threshold_score, "High", "Low")
    
    df_use <- data.frame(
      MI_value = MI_value,
      MI_group = MI_group,
      clinical_feature = clinical_feature_cur_vec,
      stringsAsFactors = FALSE
    )
    
    ## AJCC-specific handling
    if (clinical_feature_cur == "ajcc_pathologic_stage") {
      df_use$clinical_feature[df_use$clinical_feature == "Stage X"] <- NA
      df_use$clinical_feature <- ajcc_stage_numeric_map[df_use$clinical_feature]
    }
    
    if (clinical_feature_cur == "ajcc_pathologic_t") {
      df_use$clinical_feature[df_use$clinical_feature %in% c("TX", "")] <- NA
      df_use$clinical_feature[df_use$clinical_feature %in%
                                c("Tis (DCIS)", "Tis (LCIS)")] <- "Tis"
      df_use$clinical_feature <- ajcc_t_numeric_map[df_use$clinical_feature]
    }
    
    if (clinical_feature_cur == "ajcc_pathologic_n") {
      df_use$clinical_feature[df_use$clinical_feature %in% c("NX", "")] <- NA
      df_use$clinical_feature <- ajcc_n_numeric_map[df_use$clinical_feature]
    }
    
    if (clinical_feature_cur == "ajcc_pathologic_m") {
      df_use$clinical_feature[df_use$clinical_feature %in% c("MX", "")] <- NA
      df_use$clinical_feature <- ajcc_m_numeric_map[df_use$clinical_feature]
    }
    
    df_use <- na.omit(df_use)
    
    if (nrow(df_use) < 5) {
      pvalue_res_vec <- c(pvalue_res_vec, NA)
      statistic_res_vec <- c(statistic_res_vec, NA)
      next
    }
    
    if (feature_type == "categorical") {
      
      contingency_table <- table(df_use$MI_group, df_use$clinical_feature)
      if (all(dim(contingency_table) >= 2)) {
        ft <- fisher.test(
          contingency_table,
          simulate.p.value = TRUE,
          B = 1e5
        )
        pvalue_res_vec <- c(pvalue_res_vec, ft$p.value)
        statistic_res_vec <- c(statistic_res_vec, NA)
      } else {
        pvalue_res_vec <- c(pvalue_res_vec, NA)
        statistic_res_vec <- c(statistic_res_vec, NA)
      }
      
    } else {
      ##Corr
      # ct <- suppressWarnings(
      #   cor.test(
      #     df_use$MI_value,
      #     as.numeric(df_use$clinical_feature),
      #     method = "spearman"
      #   )
      # )
      # pvalue_res_vec <- c(pvalue_res_vec, ct$p.value)
      # statistic_res_vec <- c(statistic_res_vec, ct$estimate)
      
      ##Two groups (MI_group) and wilcoxon test
      clinical_feature_high <- df_use$clinical_feature[df_use$MI_group == "High"]
      clinical_feature_low  <- df_use$clinical_feature[df_use$MI_group == "Low"]
      ##One-sided test: high > low
      wt1 <- wilcox.test(
        clinical_feature_high,
        clinical_feature_low,
        alternative = "greater"
      )
      pvalue_wt1 <- wt1$p.value
      wt2 <- wilcox.test(
        clinical_feature_high,
        clinical_feature_low,
        alternative = "less"
      )
      pvalue_wt2 <- wt2$p.value
      pvalue_choose <- ifelse(pvalue_wt1 < pvalue_wt2, pvalue_wt1, pvalue_wt2)
      # statistic_res_vec <- c(statistic_res_vec,ifelse(pvalue_wt1 < pvalue_wt2, 1, -1))
      statistic_res_vec <- c(statistic_res_vec,log2(
        mean(clinical_feature_high) / mean(clinical_feature_low)
      ))
      pvalue_res_vec <- c(pvalue_res_vec, pvalue_choose)
      
     
    }
  }
  
  pvalue_res_mat[clinical_feature_cur, ] <- pvalue_res_vec
  statistic_res_mat[clinical_feature_cur, ] <- statistic_res_vec
}

names(feature_type_vec) <- clinical_feature_useful


statistic_res_mat[which(feature_type_vec == 'continuous'),]

##Select the continuous features
bool_vec <-(rowSums(pvalue_res_mat[which(feature_type_vec != 'categorical'),] < 0.05) >0) & (apply(abs(statistic_res_mat[which(feature_type_vec != 'categorical'),]),MARGIN = 1,max) > log2fc_threshold)
feature_continuous_selected <- names(bool_vec)[which(bool_vec)]

# feature_continuous_selected <- c(
#   "ajcc_pathologic_stage", "ajcc_pathologic_t","ajcc_pathologic_n",
#   "paper_hypermutated","paper_nonsilent_mutrate",
#   "paper_Fraction_genome_altered","paper_ABSOLUTE_purity"
#   
# )
##Select the categorical features
bool_vec_cat <-(rowSums(pvalue_res_mat[which(feature_type_vec == 'categorical'),] < 0.05) >0)
feature_categorical_selected <- names(bool_vec_cat)[which(bool_vec_cat)]

# feature_selected_all <- c(feature_continuous_selected, feature_categorical_selected)

##Continue with the continuous features
pvalue_res_mat_continuous_selected <- pvalue_res_mat[feature_continuous_selected, ]
statistic_res_mat_continuous_selected <- statistic_res_mat[feature_continuous_selected, ]
colnames(pvalue_res_mat_continuous_selected) <- colnames(MI_decomposition_list[[1]])
colnames(statistic_res_mat_continuous_selected) <- colnames(MI_decomposition_list[[1]])


pvalue_res_mat_continuous_selected_adj <- t(
  apply(
    pvalue_res_mat_continuous_selected,
    1,
    function(p) p.adjust(p, method = "BH")
    # function(p) p.adjust(p, method = "BY")
  )
)
pvalue_res_mat_continuous_selected_adj
rowSums((pvalue_res_mat_continuous_selected_adj < 0.05) * (statistic_res_mat_continuous_selected > 0))
rowSums((pvalue_res_mat_continuous_selected_adj < 0.05) * (statistic_res_mat_continuous_selected < 0))

p_mat <- pvalue_res_mat_continuous_selected_adj
stat_mat <- statistic_res_mat_continuous_selected

##Remove the "paper_" in pvalue_res_mat_continuous_selected's rownames
rownames(p_mat) <- gsub("paper_","", rownames(p_mat))
rownames(stat_mat) <- gsub("paper_","", rownames(stat_mat))

##Remove the "-" in MI_order
MI_order_clean <- gsub("-","", MI_order)
MI_order_clean <- rev(MI_order_clean)
p_mat <- p_mat[, MI_order_clean]
stat_mat <- stat_mat[, MI_order_clean]
row_order <-order(rowSums(log10(p_mat) * (-1)),decreasing = TRUE)
row_order_name <- rownames(p_mat)[row_order]
p_mat <- p_mat[row_order, ]
stat_mat <- stat_mat[row_order, ]

library(dplyr)
library(tidyr)
library(ggplot2)

## ---- reshape ----
df_p <- as.data.frame(p_mat)
df_p$Feature <- rownames(df_p)
df_p <- df_p %>% pivot_longer(
  cols = -Feature,
  names_to = "Index",
  values_to = "pvalue"
)

df_stat <- as.data.frame(stat_mat)
df_stat$Feature <- rownames(df_stat)
df_stat <- df_stat %>% pivot_longer(
  cols = -Feature,
  names_to = "Index",
  values_to = "stat"
)

df <- left_join(df_p, df_stat, by = c("Feature","Index"))

## ---- 保持原始顺序 ----
df$Feature <- factor(df$Feature, levels = rownames(p_mat))
df$Index   <- factor(df$Index,   levels = colnames(p_mat))

## ---- 点大小 = -log10(p)，并限制最大效果 ----
df$logp <- -log10(df$pvalue)
df$logp_ori <- df$logp
max_size_value <- 6
df$logp[df$logp > max_size_value] <- max_size_value

##Remove the "paper_" in df$Feature
df$Feature <- gsub("paper_","", df$Feature)
df$Feature <- factor(df$Feature, levels = row_order_name)
## ---- plot ----
## 先定义一个标记：显著且效果不小
df$SigStrong <- with(df, abs(stat) >= log2fc_threshold & pvalue <= 0.05)

P1 <- ggplot(df, aes(x = Index, y = Feature)) +
  # ## 1）所有点先画成浅灰色背景
  # geom_point(
  #   data = df,
  #   aes(size = logp),
  #   color = "grey90"
  # ) +
  ## 2）再把显著且|stat|>=0.1的点覆盖成蓝-白-红渐变
  geom_point(
    data = subset(df, SigStrong),
    aes(size = logp, color = stat)
  ) +
  scale_size_continuous(
    name = "-log10(p)",
    range = c(1, 10)
  ) +
  scale_color_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    name = "Statistic"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 16),
    axis.text.y = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "",
    y = ""
  ) +
  coord_flip()

P1

##Save the plot
ggsave(paste0(file_path_main, "TCGA_MIs_Continuous_Features_Bubbleplot_",MI_type,".pdf"),
       plot = P1,
       width = 6,
       height = 12)

df_filter <- df %>% filter(SigStrong == TRUE)

##Reverse the level of df_filter$Index
df_filter$Index   <- factor(df_filter$Index,   levels = rev(colnames(p_mat)))
library(ggplot2)

P2 <- ggplot(df_filter, aes(x = Index, y = Feature)) +
  geom_point(
    aes(
      size  = logp,
      color = stat > 0
    ),
    alpha = 0.9
  ) +
  scale_color_manual(
    values = c(
      "TRUE"  = "black",
      "FALSE" = "grey70"
    ),
    guide = "none"
  ) +
  scale_size_continuous(
    range = c(2, 8),
    name = expression(-log[10](p))
  ) +
  ## ★ 关键：显示所有 MI level（即使没有数据）
  scale_x_discrete(
    limits = levels(df_filter$Index),
    drop   = FALSE
  ) +
  theme_minimal(base_size = 12) +
  theme(
    ## ★ 去掉所有网格线
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  )

ggsave(
  paste0(
    file_path_main,
    "TCGA_MIs_Continuous_Features_Bubbleplot_",
    MI_type,
    "_filter.pdf"
  ),
  plot  = P2,
  width = 9,
  height = 2
)


# ##Categorical features can be ploted similarly
# 
# # feature_categorical_selected <- setdiff(feature_categorical_selected,c("paper_MSI_status", "paper_BRCA_Subtype_PAM50"))
# pvalue_res_mat_categorical_selected <- pvalue_res_mat[feature_categorical_selected, ]
# statistic_res_mat_categorical_selected <- statistic_res_mat[feature_categorical_selected, ]
# colnames(pvalue_res_mat_categorical_selected) <- colnames(MI_decomposition_list[[1]])
# colnames(statistic_res_mat_categorical_selected) <- colnames(MI_decomposition_list[[1]])
# pvalue_res_mat_categorical_selected <- pvalue_res_mat_categorical_selected[, MI_order_clean]
# statistic_res_mat_categorical_selected <- statistic_res_mat_categorical_selected[, MI_order_clean]
# 
# ##Remove the "paper_" in pvalue_res_mat_categorical_selected's rownames
# rownames(pvalue_res_mat_categorical_selected) <- gsub("paper_","", rownames(pvalue_res_mat_categorical_selected))
# 
# pvalue_res_mat_categorical_selected_adj <- t(
#   apply(
#     pvalue_res_mat_categorical_selected,
#     1,
#     function(p) p.adjust(p, method = "BH")
#   )
# )
# 
# p_mat_cat <- pvalue_res_mat_categorical_selected
# 
# 
# 
# library(dplyr)
# library(tidyr)
# library(ggplot2)
# 
# ## ---- reshape into long dataframe ----
# df_p <- as.data.frame(p_mat_cat)
# df_p$Feature <- rownames(df_p)
# df_p <- df_p %>% pivot_longer(
#   cols = -Feature,
#   names_to = "Index",
#   values_to = "pvalue"
# )
# 
# ## ---- 保持原始矩阵行列顺序 ----
# df_p$Feature <- factor(df_p$Feature, levels = rownames(p_mat_cat))
# df_p$Index   <- factor(df_p$Index,   levels = colnames(p_mat_cat))
# 
# ## ---- 点大小 = -log10(p)，并限制最大值 ----
# df_p$logp <- -log10(df_p$pvalue)
# max_size_value <- 6
# df_p$logp[df_p$logp > max_size_value] <- max_size_value
# 
# df_p$SigStrong <- with(df_p, pvalue <= 0.05)
# ## ---- bubble plot (无颜色映射) ----
# P2<-ggplot(df_p, aes(x = Index, y = Feature)) +
#   geom_point(
#     aes(size = logp),
#     color = "grey90"
#   ) +
# 
#   ## 2）显著且|stat|>=0.1 的点覆盖成黑色
#   geom_point(
#     data = subset(df_p, SigStrong),
#     aes(size = logp),
#     color = "black"
#   ) +
#   scale_size_continuous(
#     name = "-log10(p)",
#     range = c(1, 10)
#   ) +
#   theme_minimal(base_size = 12) +
#   theme(
#     axis.text.x = element_text(angle = 60, hjust = 1,size=16),
#     axis.text.y = element_text(size=16),
#     panel.grid.major = element_blank(),
#     panel.grid.minor = element_blank()
#   ) +
#   labs(
#     x = "",
#     y = ""
#     # title = "Association Bubble Plot (Categorical Features)"
#   )+
#   coord_flip()
# P2
# ##Save the plot
# ggsave(paste0(file_path_main, "TCGA_MIs_Categorical_Features_Bubbleplot_",MI_type,".pdf"),
#        plot = P2,
#        width = 9,
#        height = 12)
# 



##Load the mSiganturedc database
MI_type <- "MIdecomposition"

signature_profile_sample <- read.table("D:/CellFlowMap/Pancancer/Data/mSignatureDB/signature_profile_sample.txt", header=TRUE, sep="\t")
MSI_profile <- read.csv("D:/CellFlowMap/Pancancer/Data/MSI/ds_po.17.00073-1.csv", header=TRUE)
##Change the '-' in MSI_profile$Case.ID to '.'
MSI_profile$Case.ID <- gsub("-",".", MSI_profile$Case.ID)
Tumorfeature_profile <- read.csv("D:/CellFlowMap/Pancancer/Data/Tumorfeatures/1-s2.0-S1074761318301213-mmc2.csv", header=TRUE)
Tumorfeature_profile$TCGA.Participant.Barcode <- gsub("-",".", Tumorfeature_profile$TCGA.Participant.Barcode)

load(file=paste0(file_path_main, "MI_decomposition_list_",MI_type,".Rdata"))
load(file=paste0(file_path_main, "clinical_data_list_",MI_type,".Rdata"))

Avg_MI_cellclass_pair_heatmap_values <- read.csv(paste0(file_path_main, "Avg_MI_cellclass_pair_heatmap_values.csv"), row.names=1)
MI_order <- rownames(Avg_MI_cellclass_pair_heatmap_values)

# mutation_signature_unique <- unique(signature_profile_sample$Signature)
mutation_signature_unique <- paste("Signature.",1:30, sep="")

TCGA_project_use <- names(MI_decomposition_list)

MI_decomposition_all <- Reduce(rbind, MI_decomposition_list)

id_short <- sapply(
  strsplit(rownames(MI_decomposition_all), "-"),
  function(x) paste(x[1:3], collapse = "-")
)
id_short <- gsub("-",".", id_short)
signature_profile_sample_use <- signature_profile_sample[which(signature_profile_sample$Tumor_Sample_Barcode %in% id_short), ]
MSI_profile_use <- MSI_profile[which(MSI_profile$Case.ID %in% id_short), ]
Tumorfeature_profile_use <- Tumorfeature_profile[which(Tumorfeature_profile$TCGA.Participant.Barcode %in% id_short), ]
rownames(Tumorfeature_profile_use) <- Tumorfeature_profile_use$TCGA.Participant.Barcode
Tumorfeature_choose <- names(Tumorfeature_profile_use)[5:29]

corr_MI_mutatation <- matrix(NA, nrow=length(mutation_signature_unique), ncol=ncol(MI_decomposition_all))
rownames(corr_MI_mutatation) <- mutation_signature_unique
colnames(corr_MI_mutatation) <- colnames(MI_decomposition_all)
pval_MI_mutatation <- matrix(NA, nrow=length(mutation_signature_unique), ncol=ncol(MI_decomposition_all))
rownames(pval_MI_mutatation) <- mutation_signature_unique
colnames(pval_MI_mutatation) <- colnames(MI_decomposition_all)
corr_MI_MSI <- matrix(NA, nrow=1, ncol=ncol(MI_decomposition_all))
rownames(corr_MI_MSI) <- "MSI"
colnames(corr_MI_MSI) <- colnames(MI_decomposition_all)
pval_MI_MSI <- matrix(NA, nrow=1, ncol=ncol(MI_decomposition_all))
rownames(pval_MI_MSI) <- "MSI"
colnames(pval_MI_MSI) <- colnames(MI_decomposition_all)
corr_MI_Tumorfeature <- matrix(NA, nrow=length(Tumorfeature_choose), ncol=ncol(MI_decomposition_all))
rownames(corr_MI_Tumorfeature) <- Tumorfeature_choose
colnames(corr_MI_Tumorfeature) <- colnames(MI_decomposition_all)
pval_MI_Tumorfeature <- matrix(NA, nrow=length(Tumorfeature_choose), ncol=ncol(MI_decomposition_all))
rownames(pval_MI_Tumorfeature) <- Tumorfeature_choose
colnames(pval_MI_Tumorfeature) <-colnames(MI_decomposition_all)



##signature_profile_sample_use_subset
for (mutation_signature_cur in mutation_signature_unique){
  signature_profile_sample_use_subset <- signature_profile_sample_use[which(signature_profile_sample_use$Signature == mutation_signature_cur), ]
  rownames(signature_profile_sample_use_subset) <- signature_profile_sample_use_subset$Tumor_Sample_Barcode 
  MSI_profile_use_subset <- MSI_profile_use[which(MSI_profile_use$Case.ID %in% signature_profile_sample_use_subset$Tumor_Sample_Barcode), ]
  rownames(MSI_profile_use_subset) <- MSI_profile_use_subset$Case.ID
  id_short_common <- intersect(id_short, rownames(signature_profile_sample_use_subset))
  id_short_common_id <- which(id_short %in% id_short_common)
  id_short_common <- id_short[id_short_common_id]
  MI_decomposition_cur_subset <- MI_decomposition_all[id_short_common_id, ]
  signature_profile_sample_use_subset <- signature_profile_sample_use_subset[id_short_common, ]
  for (MI_index in 1:ncol(MI_decomposition_cur_subset)){
    MI_cur <- colnames(corr_MI_mutatation)[MI_index]
    # ##Corr
    # cor_test_result <- cor.test(MI_decomposition_cur_subset[,MI_index], signature_profile_sample_use_subset$Contribution,
    #                             # method = "pearson")
    #                             method = "spearman")
    # corr_MI_mutatation[mutation_signature_cur, MI_cur] <- cor_test_result$estimate
    # pval_MI_mutatation[mutation_signature_cur, MI_cur] <- cor_test_result$p.value
    ##Wilcoxon test
    # threshold_score <- median(MI_decomposition_cur_subset[,MI_index])
    # threshold_score <- quantile(MI_decomposition_cur_subset[,MI_index],0.75, na.rm = TRUE)
    threshold_score <- quantile(MI_decomposition_cur_subset[,MI_index],0.3, na.rm = TRUE)
    # threshold_score <- 0.5
    MI_group_cur <- ifelse(MI_decomposition_cur_subset[,MI_index] >= threshold_score, "High", "Low")
    contribution_high <- signature_profile_sample_use_subset$Contribution[which(MI_group_cur == "High")]
    contribution_low  <- signature_profile_sample_use_subset$Contribution[which(MI_group_cur == "Low")]
    ##One-sided test: high > low
    wt1 <- wilcox.test(
      contribution_high,
      contribution_low,
      alternative = "greater"
    )
    pvalue_wt1 <- wt1$p.value
    wt2 <- wilcox.test(
      contribution_high,
      contribution_low,
      alternative = "less"
    )
    pvalue_wt2 <- wt2$p.value
    pvalue_choose <- ifelse(pvalue_wt1 < pvalue_wt2,
                            pvalue_wt1,
                            pvalue_wt2)
    # corr_MI_mutatation[mutation_signature_cur, MI_cur] <- ifelse(pvalue_wt1 < pvalue_wt2, 1, -1)
    corr_MI_mutatation[mutation_signature_cur, MI_cur] <- log2(mean(contribution_high)+1e-6) - log2(mean(contribution_low)+1e-6)
    pval_MI_mutatation[mutation_signature_cur, MI_cur] <- pvalue_choose
  }
}

##MSI_profile_use_subset
id_short_common <- intersect(id_short, rownames(MSI_profile_use_subset))
id_short_common_id <- which(id_short %in% id_short_common)
id_short_common <- id_short[id_short_common_id]
MI_decomposition_cur_subset <- MI_decomposition_all[id_short_common_id, ]
MSI_profile_use_subset <- MSI_profile_use_subset[id_short_common, ]
for (MI_index in 1:ncol(MI_decomposition_cur_subset)){
  MI_cur <- colnames(corr_MI_mutatation)[MI_index]
  # ##Corr
  # cor_test_result <- cor.test(MI_decomposition_cur_subset[,MI_index], MSI_profile_use_subset$MANTIS.Score,
  #                             # method = "pearson")
  #                             method = "spearman")
  # corr_MI_MSI["MSI", MI_cur] <- cor_test_result$estimate
  # pval_MI_MSI["MSI", MI_cur] <- cor_test_result$p.value
  ##Wilcoxon test
  # threshold_score <- median(MI_decomposition_cur_subset[,MI_index])
  # threshold_score <- quantile(MI_decomposition_cur_subset[,MI_index],0.75, na.rm = TRUE)
  threshold_score <- quantile(MI_decomposition_cur_subset[,MI_index],0.3, na.rm = TRUE)
  # threshold_score <- 0.5
  MI_group_cur <- ifelse(MI_decomposition_cur_subset[,MI_index] >= threshold_score, "High", "Low")
  MANTIS_high <- MSI_profile_use_subset$MANTIS.Score[which(MI_group_cur == "High")]
  MANTIS_low  <- MSI_profile_use_subset$MANTIS.Score[which(MI_group_cur == "Low")]
  ##One-sided test: high > low
  wt1 <- wilcox.test(
    MANTIS_high,
    MANTIS_low,
    alternative = "greater"
  )
  pvalue_wt1 <- wt1$p.value
  wt2 <- wilcox.test(
    MANTIS_high,
    MANTIS_low,
    alternative = "less"
  )
  pvalue_wt2 <- wt2$p.value
  pvalue_choose <- ifelse(pvalue_wt1 < pvalue_wt2,
                          pvalue_wt1,
                          pvalue_wt2)
  # corr_MI_MSI["MSI", MI_cur] <- ifelse(pvalue_wt1 < pvalue_wt2, 1, -1)
  corr_MI_MSI["MSI", MI_cur] <- log2(mean(MANTIS_high)+1e-6) - log2(mean(MANTIS_low)+1e-6)
  pval_MI_MSI["MSI", MI_cur] <- pvalue_choose
}


##Tumorfeature_profile_use_subset
for (Tumorfeature_choose_cur in Tumorfeature_choose){
  Tumorfeature_profile_use_subset <- Tumorfeature_profile_use[, c("TCGA.Participant.Barcode",Tumorfeature_choose_cur)]
  id_short_common <- intersect(id_short, rownames(Tumorfeature_profile_use_subset))
  id_short_common_id <- which(id_short %in% id_short_common)
  id_short_common <- id_short[id_short_common_id]
  MI_decomposition_cur_subset <- MI_decomposition_all[id_short_common_id, ]
  Tumorfeature_profile_use_subset <- Tumorfeature_profile_use_subset[id_short_common, ]
  for (MI_index in 1:ncol(MI_decomposition_cur_subset)){
    MI_cur <- colnames(corr_MI_mutatation)[MI_index]
    # #Corr
    # cor_test_result <- cor.test(MI_decomposition_cur_subset[,MI_index], as.numeric(Tumorfeature_profile_use_subset[[Tumorfeature_choose_cur]]),
    #                             # method = "pearson")
    #                             method = "spearman")
    # corr_MI_Tumorfeature[Tumorfeature_choose_cur, MI_cur] <- cor_test_result$estimate
    # pval_MI_Tumorfeature[Tumorfeature_choose_cur, MI_cur] <- cor_test_result$p.value
    ##Wilcoxon test
    # threshold_score <-median(MI_decomposition_cur_subset[,MI_index])
    # threshold_score <- quantile(MI_decomposition_cur_subset[,MI_index],0.75, na.rm = TRUE)
    threshold_score <- quantile(MI_decomposition_cur_subset[,MI_index],0.3, na.rm = TRUE)
    # threshold_score <- 0.5
    MI_group_cur <- ifelse(MI_decomposition_cur_subset[,MI_index] >= threshold_score, "High", "Low")
    if (min(Tumorfeature_profile_use_subset[[Tumorfeature_choose_cur]][!is.na(Tumorfeature_profile_use_subset[[Tumorfeature_choose_cur]])]) <0){
      vector_use <- Tumorfeature_profile_use_subset[[Tumorfeature_choose_cur]] + abs(min(Tumorfeature_profile_use_subset[[Tumorfeature_choose_cur]][!is.na(Tumorfeature_profile_use_subset[[Tumorfeature_choose_cur]])])) + 1e-6
      } else {
        vector_use <- Tumorfeature_profile_use_subset[[Tumorfeature_choose_cur]]
    }
    feature_high <- vector_use[which(MI_group_cur == "High")]
    feature_low  <- vector_use[which(MI_group_cur == "Low")]
    ##One-sided test: high > low
    wt1 <- wilcox.test(
      feature_high,
      feature_low,
      alternative = "greater"
    )
    pvalue_wt1 <- wt1$p.value
    wt2 <- wilcox.test(
      feature_high,
      feature_low,
      alternative = "less"
    )
    pvalue_wt2 <- wt2$p.value
    pvalue_choose <- ifelse(pvalue_wt1 < pvalue_wt2,
                            pvalue_wt1,
                            pvalue_wt2)
    # corr_MI_Tumorfeature[Tumorfeature_choose_cur, MI_cur] <- ifelse(pvalue_wt1 < pvalue_wt2, 1, -1)
    corr_MI_Tumorfeature[Tumorfeature_choose_cur, MI_cur] <- log2(mean(feature_high[!is.na(feature_high)])+1e-6) - log2(mean(feature_low[!is.na(feature_low)])+1e-6)
    pval_MI_Tumorfeature[Tumorfeature_choose_cur, MI_cur] <- pvalue_choose
  }
}


summary(as.vector(corr_MI_mutatation))


##Extend the corr_MI_mutatation with MSI
corr_MI_mutatation <- rbind(corr_MI_mutatation, corr_MI_MSI)
pval_MI_mutatation <- rbind(pval_MI_mutatation, pval_MI_MSI)

##Extend the corr_MI_mutatation with Tumorfeature
corr_MI_mutatation <- rbind(corr_MI_mutatation, corr_MI_Tumorfeature)
pval_MI_mutatation <- rbind(pval_MI_mutatation, pval_MI_Tumorfeature)

##ADjust the pval_MI_mutatation
pval_MI_mutatation_adj <- t(
  apply(
    pval_MI_mutatation,
    1,
    function(p) p.adjust(p, method = "BH")
    # function(p) p.adjust(p, method = "BY")
  )
)

# choose_array <- (pval_MI_mutatation < 0.05) & (abs(corr_MI_mutatation) > 0.2)
# choose_array <- ((pval_MI_mutatation_adj < 0.05) & (abs(corr_MI_mutatation) > 0.5))
choose_array <- ((pval_MI_mutatation_adj < 0.05) & (abs(corr_MI_mutatation) > log2fc_threshold))

Signature_chhose <- names(rowSums(choose_array))[which(rowSums(choose_array) > 0)]
# 
# pval_MI_mutatation <- pval_MI_mutatation[Signature_chhose, ]
# corr_MI_mutatation <- corr_MI_mutatation[Signature_chhose, ]


pval_MI_mutatation_adj <- pval_MI_mutatation_adj[Signature_chhose, ]
corr_MI_mutatation <- corr_MI_mutatation[Signature_chhose, ]
p_mat <- pval_MI_mutatation_adj
stat_mat <- corr_MI_mutatation

##Remove the "paper_" in pvalue_res_mat_continuous_selected's rownames
rownames(p_mat) <- gsub("paper_","", rownames(p_mat))
rownames(stat_mat) <- gsub("paper_","", rownames(stat_mat))

##Remove the "-" in MI_order
MI_order_clean <- gsub("-","", MI_order)
MI_order_clean <- rev(MI_order_clean)
p_mat <- p_mat[, MI_order_clean]
stat_mat <- stat_mat[, MI_order_clean]
row_order <-order(rowSums(log10(p_mat) * (-1)),decreasing = TRUE)
row_order_name <- rownames(p_mat)[row_order]
p_mat <- p_mat[row_order, ]
stat_mat <- stat_mat[row_order, ]

library(dplyr)
library(tidyr)
library(ggplot2)

## ---- reshape ----
df_p <- as.data.frame(p_mat)
df_p$Feature <- rownames(df_p)
df_p <- df_p %>% pivot_longer(
  cols = -Feature,
  names_to = "Index",
  values_to = "pvalue"
)

df_stat <- as.data.frame(stat_mat)
df_stat$Feature <- rownames(df_stat)
df_stat <- df_stat %>% pivot_longer(
  cols = -Feature,
  names_to = "Index",
  values_to = "stat"
)

df <- left_join(df_p, df_stat, by = c("Feature","Index"))

## ---- 保持原始顺序 ----
df$Feature <- factor(df$Feature, levels = rownames(p_mat))
df$Index   <- factor(df$Index,   levels = colnames(p_mat))

## ---- 点大小 = -log10(p)，并限制最大效果 ----
df$logp <- -log10(df$pvalue)
df$logp_ori <- df$logp
max_size_value <- 6
df$logp[df$logp > max_size_value] <- max_size_value
# df$stat[df$stat > 0.3] <- 0.3
# df$stat[df$stat < -0.3] <- -0.3

##Assign the feature source
df$Feature_source <- ifelse(df$Feature %in% mutation_signature_unique, "mSignatureDB",
                            ifelse(df$Feature == "MSI", "MSI",
                                   "Tumor_Feature"))

##Remove the "paper_" in df$Feature（理论上前面已经去过，这里防御性再做一次）
df$Feature <- gsub("paper_","", df$Feature)
df$Feature <- factor(df$Feature, levels = row_order_name)

## ========= 按 source 重新排序 Feature：先按大类，再保留原有顺序 =========
# 当前 Feature 的顺序（此时是 row_order_name 对应顺序）
cur_levels <- levels(df$Feature)

# 三个大类的顺序
source_order <- c("mSignatureDB", "MSI", "Tumor_Feature")

# 按照 source_order，把 Feature 分成 3 块，但块内保留 cur_levels 中原来的相对顺序
new_feature_levels <- unlist(
  lapply(source_order, function(s) {
    cur_levels[df$Feature_source[match(cur_levels, as.character(df$Feature))] == s]
  })
)

# 重新设定 Feature 的因子顺序
df$Feature <- factor(df$Feature, levels = new_feature_levels)

## ========= 计算三个 source 之间竖线位置 =========
tbl_source <- table(df$Feature_source[match(levels(df$Feature), as.character(df$Feature))])
tbl_source <- tbl_source[source_order]  # 保证顺序一致
cum_len <- cumsum(tbl_source)
vline_pos <- c(0,cum_len) + 0.5  # 每块之间的分界位置

library(ggtext)

## ========= Feature_source 颜色（给列名上色） =========
feature_source_cols <- c(
  "mSignatureDB"  = "#E41A1C",
  "MSI"           = "#377EB8",
  "Tumor_Feature" = "#4DAF4A"
)

## 每个 Feature → source 的映射（用于 axis label 上色）
feature2source <- df %>%
  dplyr::select(Feature, Feature_source) %>%
  dplyr::distinct() %>%
  tibble::deframe()

## ========= 为颜色创建一个新变量：stat_col =========
df$stat_col <- df$stat
# df$stat_col[(abs(df$stat_col) < 0.1) | (df$pvalue > 0.05)] <- 0  # (-0.1, 0.1) 都映射到 0
df$stat_col[(abs(df$stat_col) < log2fc_threshold) | (df$pvalue > 0.05)] <- 0  # (-0.1, 0.1) 都映射到 0

## ========= 交换坐标（真正让 Feature 成为列） =========
P3 <- ggplot(df, aes(x = Feature, y = Index)) +
  geom_point(aes(size = logp, color = stat_col)) +
  geom_vline(xintercept = vline_pos, color = "black", linewidth = 1) +
  scale_size_continuous(
    name = "-log10(p)",
    range = c(1, 10)
  ) +
  scale_color_gradient2(
    low = "blue",
    mid = "grey90",   # 中间是浅灰
    high = "red",
    midpoint = 0,
    name = "Statistic",
    limits = c(-max(abs(df$stat)), max(abs(df$stat)))
  ) +
  ## ========= 给 “列 Feature 名字” 上色 =========
scale_x_discrete(
  labels = function(x) {
    sapply(x, function(f) {
      src <- feature2source[[as.character(f)]]
      col <- feature_source_cols[[src]]
      sprintf("<span style='color:%s'>%s</span>", col, f)
    })
  }
) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = ggtext::element_markdown(angle = 45, hjust = 1, size = 16),
    axis.text.y = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  labs(x = "", y = "")

P3

ggsave(
  paste0(file_path_main, "TCGA_MIs_Mutation_Signature_Bubbleplot_", MI_type, ".pdf"),
  plot = P3,
  width = 25,
  height = 12
)



df_filter2 <- df %>% filter((abs(stat) >= log2fc_threshold) & (pvalue <= 0.05))
##Remove the "stat_col" column in df_filter2
df_filter2$stat_col <- NULL

##Remove the "SigStrong" column in df_filter
df_filter$SigStrong <- NULL
df_filter$Feature_source <- ifelse(df_filter$Feature%in% c("ABSOLUTE_purity","ajcc_pathologic_t", "ajcc_pathologic_n", "ajcc_pathologic_stage"), "Tumor progression", "Genome instability")

##
df_filter_all <- rbind(df_filter, df_filter2)

df_filter_all$Index   <- factor(df_filter_all$Index,   levels = rev(colnames(p_mat)))

df_filter_all$stat_abs <- abs(df_filter_all$stat)


##Select the featues in df_filter_all
feature_choose <- unique(df_filter_all$Feature)
# feature_choose <- c(
#   "ABSOLUTE_purity",
#   "ajcc_pathologic_t",
#   "Fraction_genome_altered",
#   "hypermutated",
#   "nonsilent_mutrate",
#   "Leukocyte.Fraction",
#   "Proliferation",
#   "Lymphocyte.Infiltration.Signature.Score",
#   "TGF.beta.Response",
#   "TCR.Shannon",
#   "TCR.Richness",
#   "IFN.gamma.Response",
#   "Wound.Healing",
#   "BCR.Richness",
#   "BCR.Shannon",
#   "Number.of.Segments",
#   "TIL.Regional.Fraction",
#   "Fraction.Altered",
#   "Intratumor.Heterogeneity",
#   "SNV.Neoantigens",
#   "Homologous.Recombination.Defects",
#   "Nonsilent.Mutation.Rate",
#   "CTA.Score",
#   "Silent.Mutation.Rate",
#   "TCR.Evenness",
#   "Signature.13"
# )

## ===============================
## 1. 预处理：cap size
## ===============================
library(dplyr)
library(tidyr)
library(ggplot2)
library(tibble)

## =========================================================
## 0. cap size（与你之前一致）
## =========================================================
# df_filter_all$stat_abs_cap <- pmin(df_filter_all$stat_abs, 0.5)
df_filter_all$stat_abs_cap <- df_filter_all$stat_abs

## 确保 Index 是 factor，levels 是 MI 的顺序
mi_levels <- levels(df_filter_all$Index)

## =========================================================
# ## 1. Feature × MI 的 stat 矩阵
# ## =========================================================
# stat_mat <- df_filter_all %>%
#   select(Feature, Index, stat) %>%
#   pivot_wider(
#     names_from  = Index,
#     values_from = stat
#   ) %>%
#   column_to_rownames("Feature") %>%
#   as.matrix()
# ##Fill NA with 0
# stat_mat[is.na(stat_mat)] <- 0
# 
# ##Calculate the correlation matrix between features
# cor_mat <- cor(t(stat_mat), method = "spearman")
# library(igraph)
# 
# ## 1. 准备相关矩阵（去掉对角线）
# cor_mat_use <- cor_mat
# diag(cor_mat_use) <- 0
# 
# ## 2. 构建邻接矩阵：cor > 0.5 视为有边
# adj_mat <- cor_mat_use > 0.95
# 
# ## 确保是对称的
# adj_mat <- adj_mat | t(adj_mat)
# 
# ## 3. 构建无向图
# g <- graph_from_adjacency_matrix(
#   adj_mat,
#   mode = "undirected",
#   diag = FALSE
# )
# 
# ## 4. 找连通分量
# comp <- components(g)
# 
# ## 5. 每个 feature 对应的 cluster id
# feature_clusters <- data.frame(
#   Feature = names(comp$membership),
#   Cluster = comp$membership,
#   stringsAsFactors = FALSE
# )
# 
# feature_clusters
# set.seed(123)
# 
# feature_one_per_cluster <- feature_clusters %>%
#   group_by(Cluster) %>%
#   slice_sample(n = 1) %>%
#   ungroup()
# 
# feature_one_per_cluster
# 
# 
# ##
# df_filter_all <- df_filter_all %>%
#   filter(Feature %in% feature_one_per_cluster$Feature)
# # stat_mat <- stat_mat[feature_one_per_cluster$Feature, ]
# 
# ####
# stat_mat <- stat_mat[, rev(mi_levels)]
# 
# stat_mat <- abs(stat_mat)
# ## =========================================================
# ## 2. 按列做 z-score normalization
# ## =========================================================
# stat_mat_z <- apply(
#   stat_mat,
#   2,
#   # 1,
#   function(x) {
#     if (all(is.na(x))) return(x)
#     (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
#   }
# )
# stat_mat_z <- as.matrix(stat_mat_z)
# # stat_mat_z <- t(stat_mat_z)
# 
# ## =========================================================
# ## 3. 贪心（greedy）按 MI 顺序选 Feature
# ## =========================================================
# remaining_features <- rownames(stat_mat_z)
# ordered_features   <- c()
# 
# for (mi in mi_levels) {
#   
#   if (!mi %in% colnames(stat_mat_z)) next
#   if (length(remaining_features) == 0) break
#   
#   zvals <- stat_mat_z[remaining_features, mi]
#   
#   if (all(is.na(zvals))) next
#   
#   ## 找 |z| 最大的 Feature（可以一次拿多个）
#   max_abs_z <- max(abs(zvals), na.rm = TRUE)
#   sel_feats <- names(zvals)[abs(zvals) == max_abs_z]
#   
#   ## 记录顺序
#   ordered_features <- c(ordered_features, sel_feats)
#   
#   ## 从剩余中移除
#   remaining_features <- setdiff(remaining_features, sel_feats)
# }
# 
# ## 如果还有没被选中的 Feature，接在最后（保持原顺序）
# ordered_features <- c(ordered_features, remaining_features)
# 
# # ordered_features <- c( "TGF.beta.Response","ABSOLUTE_purity",
# #                        "Macrophage.Regulation","Fraction_genome_altered",
# #                        "Wound.Healing","Silent.Mutation.Rate", "Signature.13",
# #                        "TIL.Regional.Fraction","CTA.Score", "TCR.Evenness",
# #                        "nonsilent_mutrate",
# #                        "hypermutated","Intratumor.Heterogeneity",
# #                        "ajcc_pathologic_t"
# #                        
# #                        )
# ordered_features <- rev(ordered_features)
# 
# ## =========================================================
# ## 4. 重新设置 Feature factor 顺序
# ## =========================================================
# df_filter_all$Feature <- factor(
#   df_filter_all$Feature,
#   levels = ordered_features
# )


df_filter_all <- df_filter_all[!(df_filter_all$Feature %in%c("mRNA_cluster","RPPA_cluster","cna_cluster_k4","ajcc_pathologic_n") ),]
# df_filter_all$Feature <- factor(
#   df_filter_all$Feature,
#   levels = rev(feature_choose)
# )

Feature_unique <- unique(df_filter_all$Feature)
Index_unique <- unique(df_filter_all$Index)
stat_matrix <- matrix(0, nrow=length(Feature_unique), ncol=length(Index_unique))
for (i in 1:nrow(df_filter_all)){
  feature_cur <- df_filter_all$Feature[i]
  index_cur <- df_filter_all$Index[i]
  stat_cur <- df_filter_all$stat[i]
  stat_matrix[which(Feature_unique == feature_cur), which(Index_unique == index_cur)] <- stat_cur
}
rownames(stat_matrix) <- Feature_unique
colnames(stat_matrix) <- Index_unique

stat_matrix <- stat_matrix[, (mi_levels)]
# 
# # stat_matrix: matrix 或 data.frame（行=feature，列=MI）
# m <- as.matrix(stat_matrix)
# 
# # 每列 rank（降序：值大 rank 小）
# rank_mat <- apply(m, 2, function(x) rank(-x, ties.method = "average"))
# 
# # 每行平均 rank
# mean_rank <- rowMeans(rank_mat, na.rm = TRUE)
# 
# # 行排序后的矩阵
# stat_matrix_sorted <- m[order(mean_rank), ]

Feature_corr_mat <- cor(t(stat_matrix))

row_index_sort <-c()
for (j in 1:ncol(stat_matrix)){
  stat_col_cur <- stat_matrix[, j]
  # stat_col_order <- order(-stat_col_cur)
  stat_col_order <- order(-abs(stat_col_cur))
  stat_col_order <- setdiff(stat_col_order,row_index_sort)
  if (length(stat_col_order) >0){
    row_index_sort <- c(row_index_sort,stat_col_order[1]) 
  }
}
row_index_sort_other <- setdiff(1:nrow(stat_matrix), row_index_sort)
assign_index <- apply(Feature_corr_mat[row_index_sort_other,row_index_sort],MARGIN = 1,which.max)
assign_index_tran <- row_index_sort[assign_index]

row_index_sort_final <-c()
for (k in row_index_sort){
  if (k %in% assign_index_tran){
    assigned_features <- which(assign_index_tran == k)
    row_index_sort_final <- c(row_index_sort_final, k, row_index_sort_other[assigned_features])
  } else {
    row_index_sort_final <- c(row_index_sort_final, k)
  }
}

Feature_name_show <- rownames(stat_matrix)[row_index_sort_final]
Feature_name_show <-setdiff(Feature_name_show,c("Aneuploidy.Score",
                                                "Signature.7","Signature.11","Signature.5",
                                                "Signature.1","Signature.15",
                                                "Signature.12","Signature.8","Signature.28","Signature.30",
                                                "Homologous.Recombination.Defects","Number.of.Segments","PURITY..ABSOLUTE.","Fraction.Altered",
                                                "Leukocyte.Fraction","TIL.Regional.Fraction","TCR.Richness","BCR.Richness",
                                                "CTA.Score","Silent.Mutation.Rate","Intratumor.Heterogeneity",
                                                "SNV.Neoantigens",
                                                "Stromal.Fraction","Signature.13","Signature.17","Signature.16","Signature.4"))

Feature_name_show <- c("Signature.2",
                       "ABSOLUTE_purity",
                       "Fraction_genome_altered","Proliferation","Nonsilent.Mutation.Rate","Signature.10",
                       "IFN.gamma.Response","TCR.Shannon","Macrophage.Regulation","Lymphocyte.Infiltration.Signature.Score","Signature.14",
                       "Signature.6",
                       "TGF.beta.Response","Indel.Neoantigens",
                       "BCR.Shannon",
                       "Signature.3",
                       "Signature.22"
                       )
# ##
df_filter_all <- df_filter_all[which(df_filter_all$Feature %in%Feature_name_show),]
df_filter_all$Feature <- factor(
  df_filter_all$Feature,
  levels = rev(Feature_name_show)
)


## =========================================================
## 5. Bubble plot（与你之前一致）
P2 <- ggplot(df_filter_all, aes(x = Index, y = Feature)) +
  geom_point(
    aes(
      size  = stat_abs_cap,
      color = stat > 0
    ),
    alpha = 0.9
  ) +
  scale_color_manual(
    values = c(
      "TRUE"  = "black",
      "FALSE" = "grey70"
    ),
    guide = "none"
  ) +
  scale_size_continuous(
    # limits = c(0, 0.5),
    # range  = c(2, 8),
    range  = c(3, 8),
    name   = "|stat| (capped at 0.5)"
  ) +
  scale_x_discrete(
    limits = mi_levels,
    drop   = FALSE
  ) +
  theme_minimal(base_size = 12) +
  theme(
    ## 去掉网格线
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    ## 坐标轴文字
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    
    ## ★ 面板边框（坐标区域）
    panel.border = element_rect(
      colour = "black",
      fill   = NA,
      linewidth = 0.8
    ),
    
    ## ★ 整张图外边框
    plot.background = element_rect(
      colour = "black",
      fill   = NA,
      linewidth = 0.8
    )
  )

ggsave(
  paste0(
    file_path_main,
    "TCGA_MIs_Continuous_Features_Bubbleplot_",
    MI_type,
    "_filter.pdf"
  ),
  plot  = P2,
  # width = 9,
  width = 10,
  # height = 5.1
  height = 4.7
  # width = 11,
  # height = 12
)




##Do the MI composition analysis for each TCGA project
# load the tissue_type_df
tissue_type_df <- read.csv(file=paste0(file_path_main, "TCGA_tissue_type_annotation_",MI_type,".csv"), header=TRUE)
rownames(tissue_type_df) <- tissue_type_df$X
##Remove the "X" column
tissue_type_df$X <- NULL
##Remve the character before "." in tissue_type_df's rownames
rownames(tissue_type_df) <- sapply(rownames(tissue_type_df), function(x) unlist(strsplit(x, split="\\."))[2])

##LOad the MI_decomposition_list (save(MI_decomposition_list, file=paste0(file_path_main,"MI_decomposition_list_",MI_type,".Rdata")))
load(file=paste0(file_path_main,"MI_decomposition_list_",MI_type,".Rdata"))
MI_decomposition_df <- Reduce(rbind, MI_decomposition_list)
# rownames(MI_decomposition_df) <- rownames(tissue_type_df)

##Tissue_type_use
Tissue_type_use <- "type"
Tissue_type_unique <- unique(tissue_type_df[[Tissue_type_use]])



MI_composition_cur_df_list <- list()
for (Tissue_type_cur in Tissue_type_unique){
  sample_idx_cur <- which(tissue_type_df[[Tissue_type_use]] == Tissue_type_cur)
  MI_decomposition_df_cur <- MI_decomposition_df[sample_idx_cur, ]
  MI_composition_cur <- colMeans(MI_decomposition_df_cur)
  # MI_composition_cur <-prop.table(table(apply(MI_decomposition_df_cur,MARGIN = 1,FUN = function(x){which.max(x)})))
  # names(MI_composition_cur) <- paste("MI", names(MI_composition_cur), sep="")
  ##Check if any MI is missing, is missing, set it to 0
  MI_missing <- setdiff(colnames(MI_decomposition_df_cur), names(MI_composition_cur))
  if (length(MI_missing) > 0){
    for (MI_missing_cur in MI_missing){
      MI_composition_cur[MI_missing_cur] <- 0
    }
  }
  MI_composition_cur <- MI_composition_cur[colnames(MI_decomposition_df_cur)]
  MI_composition_cur_df <- data.frame(
    MI = names(MI_composition_cur),
    Composition = as.vector(MI_composition_cur)
  )
  MI_composition_cur_df$Tissue_type <- Tissue_type_cur
  MI_composition_cur_df_list[[Tissue_type_cur]] <- MI_composition_cur_df
}
MI_composition_cur_df_all <- Reduce(rbind, MI_composition_cur_df_list)

MI_composition_cur_df_all_tumor <- MI_composition_cur_df_all[which(MI_composition_cur_df_all$Tissue_type == "Tumor"), ]
MI_composition_cur_df_all_normal <- MI_composition_cur_df_all[which(MI_composition_cur_df_all$Tissue_type == "Normal"), ]
print(MI_composition_cur_df_all[which(MI_composition_cur_df_all$Tissue_type == "Tumor"), ])
print(MI_composition_cur_df_all[which(MI_composition_cur_df_all$Tissue_type == "Normal"), ])

aaaa<-cbind(
  MI_composition_cur_df_all_tumor$Composition,
  MI_composition_cur_df_all_normal$Composition
)
rownames(aaaa) <- MI_composition_cur_df_all_tumor$MI
aaaa

## ===============================
## 1. Load packages
## ===============================
library(ggplot2)
library(reshape2)
library(dplyr)

## ===============================
## 2. Prepare data
## ===============================

# 确保 tissue_type_df$type 顺序与 MI_decomposition_df 行一致
stopifnot(nrow(MI_decomposition_df) == length(tissue_type_df$type))

# 加入 tissue type
MI_df <- MI_decomposition_df %>%
  as.data.frame() %>%
  mutate(
    sample = rownames(MI_decomposition_df),
    tissue_type = tissue_type_df$type
  )

## ===============================
## 3. Wide → Long format
## ===============================
MI_long <- melt(
  MI_df,
  id.vars = c("sample", "tissue_type"),
  variable.name = "MI",
  value.name = "MI_score"
)

## ===============================
## 4. Boxplot with facet
## ===============================
p <- ggplot(
  MI_long,
  aes(x = tissue_type, y = MI_score, fill = tissue_type)
) +
  geom_boxplot(
    outlier.shape = NA,
    width = 0.6,
    alpha = 0.8
  ) +
  geom_jitter(
    width = 0.15,
    size = 0.6,
    alpha = 0.4
  ) +
  facet_wrap(~ MI, scales = "free_y", ncol = 5) +
  scale_fill_manual(
    values = c(
      "Tumor"  = "#E41A1C",
      "Normal" = "#377EB8"
    )
  ) +
  theme_bw(base_size = 12) +
  theme(
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "",
    y = "MI score"
  )

print(p)
ggsave(
  paste0(file_path_main, "TCGA_MIs_Boxplot_by_Tissue_Type_", MI_type, ".pdf"),
  plot  = p,
  width = 12,
  height = 10
)