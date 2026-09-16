# Shared saved-result plotting and command-line paths for pan-cancer analyses.
# Scientific fitting remains in the three analysis scripts.

plot_tcga_summaries <- function(cache_dir, output_dir, TCGA_project_list) {
  # -----------------------------
  # Summarize sample-level MI abundance by TCGA cohort
  # -----------------------------
  # Reload exported estimates so cohort summaries use the saved, clipped values.
  MI_decomposition_list <- list()
  for (project_cur in TCGA_project_list) {
    MI_decomposition_list[[project_cur]] <- read.csv(
      file.path(cache_dir, paste0("MI_decomposition_", project_cur, ".csv")),
      row.names = 1
    )
  }
  # Compute each MI's mean abundance across the retained tumor samples in each cohort.
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

  write.csv(
    Mean_MI_Intensity_cluster,
    file.path(output_dir, "Mean_MI_Intensity_cluster_tumor_only.csv")
  )

  library(pheatmap)

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
    filename = file.path(output_dir, "Mean_MI_Intensity_cluster_heatmap_tumor_only.pdf"),
    width = 3.2,
    height = 4.2
  )

  # Selected MI subset for the cohort heatmap and comparison below.
  Mean_MI_Intensity_cluster_sub <- Mean_MI_Intensity_cluster[c("MI2","MI10","MI8","MI3","MI4"),]
  write.csv(
    Mean_MI_Intensity_cluster_sub,
    file.path(output_dir, "Mean_MI_Intensity_cluster_selected_tumorMI_tumor_only.csv")
  )


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
    filename = file.path(output_dir, "Mean_MI_Intensity_cluster_heatmap_selected_tumorMI_tumor_only.pdf"),
    width = 3.2,
    height = 4.2
  )


  # -----------------------------
  # Compare selected versus other MI-cohort pairs
  # -----------------------------
  # Each observation is a cohort mean for one of the five selected MIs.
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


  ## -----------------------------
  ## 2. Label Selected vs Others
  ## -----------------------------
  df_long$Group <- "Others"

  for (mi in names(selected_list)) {
    df_long$Group[df_long$MI == mi & df_long$TumorType %in% selected_list[[mi]]] <- "Selected"
  }

  df_long$Group <- factor(df_long$Group, levels = c("Selected", "Others"))


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

  write.csv(
    df_long,
    file.path(output_dir, "Selected_vs_Others_boxplot_source_tumor_only.csv"),
    row.names = FALSE
  )

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
  ## 5. Plot cohort means with the Wilcoxon annotation
  ## -----------------------------
  # Point jitter uses the current R random-number state.
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
  ## 6. Export the comparison as a Cairo PDF
  ## -----------------------------
  ggsave(
    filename = file.path(output_dir, "Selected_vs_Others_boxplot_tumor_only.pdf"),
    plot = p,
    width = 3.2,
    height = 4.2,
    device = cairo_pdf
  )


}

align_observed_mi <- function(MI_hat, MI_mean_df) {
  # Match MI1, MI-1, and MI_1 by their identifiers, never by column position.
  canonical_mi <- function(x) toupper(gsub("[^A-Za-z0-9]", "", x))
  projected_ids <- canonical_mi(colnames(MI_hat))
  observed_ids <- canonical_mi(colnames(MI_mean_df))
  observed_mi_ids <- observed_ids[grepl("^MI[0-9]+$", observed_ids)]
  if (anyDuplicated(projected_ids) || anyDuplicated(observed_mi_ids) ||
      !all(grepl("^MI[0-9]+$", projected_ids))) {
    stop("Ambiguous MI identifiers in projected or observed sub-slice matrices.")
  }
  observed_index <- match(projected_ids, observed_ids)
  if (anyNA(observed_index)) stop("Observed sub-slice matrix is missing projected MI identifiers.")
  MI_mean_df_entrie <- MI_mean_df[, observed_index, drop = FALSE]
  if (!all(vapply(MI_mean_df_entrie, is.numeric, logical(1)))) {
    stop("Observed MI columns must be numeric.")
  }
  colnames(MI_mean_df_entrie) <- colnames(MI_hat)

  MI_mean_df_entrie
}

plot_pseudobulk_recovery <- function(MI_hat, MI_mean_df_entrie, output_dir, mi_to_plot = "MI-4") {
  canonical_mi <- function(x) toupper(gsub("[^A-Za-z0-9]", "", x))
  projected_ids <- canonical_mi(colnames(MI_hat))
  # The joint fit includes all MIs; only the recovery panel selects MI-4.
  mi_index <- match(canonical_mi(mi_to_plot), projected_ids)
  if (length(mi_index) != 1L || is.na(mi_index)) {
    stop("Requested MI is absent from the pseudo-bulk projection: ", mi_to_plot)
  }
  MI_hat_flat <- as.numeric(MI_hat[, mi_index])
  MI_mean_flat <- as.numeric(MI_mean_df_entrie[, mi_index])

  n_groups <- 3

  # Output PDF path
  pdf_output_path <- file.path(
    output_dir,
    paste0("S26b_", canonical_mi(mi_to_plot), "_pseudobulk_recovery_boxplot.pdf")
  )

  # ------------------------------------------------------------
  # Combine and remove NA / Inf
  # ------------------------------------------------------------
  plot_df <- data.frame(
    sample = rownames(MI_hat),
    MI = canonical_mi(mi_to_plot),
    MI_mean = MI_mean_flat,
    MI_hat = MI_hat_flat
  )

  plot_df <- plot_df[
    is.finite(plot_df$MI_mean) & is.finite(plot_df$MI_hat),
  ]

  # ------------------------------------------------------------
  # Quantile-based grouping
  # ------------------------------------------------------------
  breaks_use <- unique(quantile(
    plot_df$MI_mean,
    probs = seq(0, 1, length.out = n_groups + 1),
    na.rm = TRUE
  ))

  n_groups_actual <- length(breaks_use) - 1

  if (n_groups_actual < 2) {
    stop("Too few unique quantile breaks. Try reducing n_groups.")
  }

  if (n_groups_actual != 3) {
    stop("This plotting code is customized for exactly 3 groups.")
  }

  plot_df$MI_mean_group <- cut(
    plot_df$MI_mean,
    breaks = breaks_use,
    include.lowest = TRUE,
    right = TRUE,
    labels = c("Low", "Intermediate", "High")
  )

  plot_df$MI_mean_group <- factor(
    plot_df$MI_mean_group,
    levels = c("Low", "Intermediate", "High")
  )

  # Check group size
  print(table(plot_df$MI_mean_group))

  # ------------------------------------------------------------
  # Helper function for p-value formatting
  # ------------------------------------------------------------
  format_pvalue <- function(p) {
    if (is.na(p)) {
      return("P = NA")
    } else if (p < 2.2e-16) {
      return("P < 2.2e-16")
    } else if (p < 0.001) {
      return(paste0("P = ", formatC(p, format = "e", digits = 2)))
    } else {
      return(paste0("P = ", signif(p, 3)))
    }
  }

  # ------------------------------------------------------------
  # Pairwise adjacent Wilcoxon rank-sum tests
  # ------------------------------------------------------------
  group_levels <- levels(plot_df$MI_mean_group)

  wilcox_adjacent_df <- data.frame(
    group1 = character(0),
    group2 = character(0),
    p_value = numeric(0),
    p_label = character(0),
    stringsAsFactors = FALSE
  )

  for (k in 1:2) {
    g1 <- group_levels[k]
    g2 <- group_levels[k + 1]
  
    df_pair <- plot_df[
      plot_df$MI_mean_group %in% c(g1, g2),
      ,
      drop = FALSE
    ]
  
    df_pair$MI_mean_group <- droplevels(df_pair$MI_mean_group)
  
    wt <- suppressWarnings(
      wilcox.test(
        MI_hat ~ MI_mean_group,
        data = df_pair,
        alternative = "two.sided",
        exact = FALSE
      )
    )
  
    wilcox_adjacent_df <- rbind(
      wilcox_adjacent_df,
      data.frame(
        group1 = g1,
        group2 = g2,
        p_value = wt$p.value,
        p_label = format_pvalue(wt$p.value),
        stringsAsFactors = FALSE
      )
    )
  }

  print(wilcox_adjacent_df)
  write.csv(plot_df, file.path(output_dir, "S26b_recovery_source.csv"), row.names = FALSE)
  write.csv(wilcox_adjacent_df, file.path(output_dir, "S26b_adjacent_tertile_Wilcoxon.csv"), row.names = FALSE)
  write.csv(data.frame(probability = seq(0, 1, length.out = n_groups + 1),
                       observed_MI_cutoff = breaks_use),
            file.path(output_dir, "S26b_observed_MI_tertile_cutoffs.csv"), row.names = FALSE)

  # ------------------------------------------------------------
  # Y-axis settings
  # ------------------------------------------------------------
  y_min <- 0
  y_axis_upper <- 0.42

  if (max(plot_df$MI_hat, na.rm = TRUE) > y_axis_upper) {
    warning("Some MI_hat values are larger than y_axis_upper and will be clipped in the plot.")
  }

  ylim_use <- c(y_min, y_axis_upper)

  # ------------------------------------------------------------
  # Open vector PDF device
  # useDingbats = FALSE makes symbols/text easier to edit in Illustrator
  # ------------------------------------------------------------
  pdf(
    file = pdf_output_path,
    width = 3.35,      # ~85 mm, Nature single-column width
    height = 3.10,
    family = "Helvetica",
    useDingbats = FALSE,
    bg = "white"
  )

  # ------------------------------------------------------------
  # Nature-style plotting parameters
  # ------------------------------------------------------------
  old_par <- par(no.readonly = TRUE)

  par(
    family = "Helvetica",
    bty = "l",
    las = 1,
    cex = 1,
    cex.axis = 0.75,
    cex.lab = 0.85,
    font.axis = 1,
    font.lab = 1,
    lwd = 0.7,
    lend = "butt",
    mar = c(3.6, 3.7, 0.8, 0.6),
    mgp = c(2.1, 0.55, 0),
    tck = -0.018
  )

  # ------------------------------------------------------------
  # Boxplot
  # ------------------------------------------------------------
  boxplot(
    MI_hat ~ MI_mean_group,
    data = plot_df,
    xlab = paste0("Observed sub-slice-average\n", mi_to_plot, " abundance"),
    ylab = paste0("Projected pseudo-bulk\n", mi_to_plot, " abundance"),
    main = "",
    las = 1,
    outline = FALSE,
    ylim = ylim_use,
    boxwex = 0.52,
    staplewex = 0.45,
    whisklty = 1,
    whisklwd = 0.7,
    staplelwd = 0.7,
    medlwd = 1.0,
    boxlwd = 0.7,
    col = "grey90",
    border = "black",
    frame.plot = FALSE
  )

  axis(
    side = 2,
    at = seq(0, 0.4, by = 0.1),
    labels = seq(0, 0.4, by = 0.1),
    las = 1,
    cex.axis = 0.75,
    lwd = 0.7,
    lwd.ticks = 0.7
  )

  # ------------------------------------------------------------
  # Add Wilcoxon p-value brackets at fixed y positions
  # Low vs Intermediate: y = 0.38
  # Intermediate vs High: y = 0.40
  # ------------------------------------------------------------
  bracket_tick <- 0.006
  text_gap <- 0.004
  bracket_lwd <- 0.7
  p_cex <- 0.68

  # ---- Low vs Intermediate ----
  x1 <- 1
  x2 <- 2
  y_bracket <- 0.38

  segments(
    x0 = x1,
    y0 = y_bracket - bracket_tick,
    x1 = x1,
    y1 = y_bracket,
    lwd = bracket_lwd
  )
  segments(
    x0 = x1,
    y0 = y_bracket,
    x1 = x2,
    y1 = y_bracket,
    lwd = bracket_lwd
  )
  segments(
    x0 = x2,
    y0 = y_bracket,
    x1 = x2,
    y1 = y_bracket - bracket_tick,
    lwd = bracket_lwd
  )

  text(
    x = (x1 + x2) / 2,
    y = y_bracket + text_gap,
    labels = wilcox_adjacent_df$p_label[1],
    cex = p_cex,
    adj = c(0.5, 0)
  )

  # ---- Intermediate vs High ----
  x1 <- 2
  x2 <- 3
  y_bracket <- 0.40

  segments(
    x0 = x1,
    y0 = y_bracket - bracket_tick,
    x1 = x1,
    y1 = y_bracket,
    lwd = bracket_lwd
  )
  segments(
    x0 = x1,
    y0 = y_bracket,
    x1 = x2,
    y1 = y_bracket,
    lwd = bracket_lwd
  )
  segments(
    x0 = x2,
    y0 = y_bracket,
    x1 = x2,
    y1 = y_bracket - bracket_tick,
    lwd = bracket_lwd
  )

  text(
    x = (x1 + x2) / 2,
    y = y_bracket + text_gap,
    labels = wilcox_adjacent_df$p_label[2],
    cex = p_cex,
    adj = c(0.5, 0)
  )

  par(old_par)
  dev.off()

  cat("Saved Nature-style editable PDF to: ", pdf_output_path, "\n", sep = "")
  invisible(list(projected = MI_hat, observed = MI_mean_df_entrie,
                 source = plot_df, tests = wilcox_adjacent_df, pdf = pdf_output_path))
}

plot_km_nature <- function(df_surv, project_cur, MI_cur, outdir,
                           adj_hr = NA_real_, adj_p = NA_real_) {

  ## basic checks
  if (nrow(df_surv) < 6) return(NULL)
  if (!all(c("time", "status", "group") %in% colnames(df_surv))) return(NULL)
  if (length(unique(df_surv$group[!is.na(df_surv$group)])) < 2) return(NULL)
  if (sum(df_surv$status == 1, na.rm = TRUE) < 3) return(NULL)

  ## Three-group order
  df_surv$group <- factor(df_surv$group, levels = c("Low", "Mid", "High"))
  df_surv <- df_surv[!is.na(df_surv$group), , drop = FALSE]
  if (length(unique(df_surv$group[!is.na(df_surv$group)])) < 2) return(NULL)

  ## KM fit
  fit_km <- try(survfit(Surv(time, status) ~ group, data = df_surv), silent = TRUE)
  if (inherits(fit_km, "try-error")) return(NULL)

  ## log-rank across available groups
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
    "Adj. HR (per 1 SD MI strength) = NA"
  } else {
    paste0("Adj. HR (per 1 SD MI strength) = ", formatC(adj_hr, format = "f", digits = 2))
  }

  wald_label <- if (is.na(adj_p)) {
    "Adj. Wald P = NA"
  } else if (adj_p < 0.001) {
    "Adj. Wald P < 0.001"
  } else {
    paste0("Adj. Wald P = ", signif(adj_p, 3))
  }

  stat_label <- paste(logrank_label, hr_label, wald_label, sep = "\n")

  ## colors: Low -> yellow, Mid -> orange, High -> dark red
  km_colors <- c(
    "Low"  = "#FFB33F",
    "Mid"  = "#FF4400",
    "High" = "#C00707"
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
  x_max_plot <- max_time * 1.12

  ## sample size per group
  n_group <- table(df_surv$group)

  ## get last point of each KM curve for n-label annotation
  curve_df <- survminer::surv_summary(fit_km, data = df_surv)
  curve_end <- do.call(
    rbind,
    lapply(split(curve_df, curve_df$strata), function(dd) dd[nrow(dd), , drop = FALSE])
  )

  curve_end$group <- sub("^group=", "", curve_end$strata)
  curve_end$group <- factor(curve_end$group, levels = c("Low", "Mid", "High"))
  curve_end$label <- paste0("n=", as.integer(n_group[as.character(curve_end$group)]))
  curve_end$x_label <- pmin(curve_end$time + max_time * 0.04, x_max_plot * 0.98)
  curve_end$y_label <- curve_end$surv

  ## avoid overlapping n labels
  if (nrow(curve_end) >= 2) {
    curve_end <- curve_end[order(curve_end$y_label), , drop = FALSE]
    min_gap <- 0.055

    for (ii in 2:nrow(curve_end)) {
      if (!is.na(curve_end$y_label[ii]) &&
          !is.na(curve_end$y_label[ii - 1]) &&
          (curve_end$y_label[ii] - curve_end$y_label[ii - 1]) < min_gap) {
        curve_end$y_label[ii] <- curve_end$y_label[ii - 1] + min_gap
      }
    }

    curve_end$y_label <- pmax(pmin(curve_end$y_label, 0.98), 0.04)
  }

  ## make plot
  g <- ggsurvplot(
    fit_km,
    data = df_surv,
    conf.int = FALSE,
    censor = TRUE,
    censor.shape = 124,
    censor.size = 1.8,
    risk.table = FALSE,
    pval = FALSE,
    palette = km_colors,
    legend.title = NULL,
    legend.labs = c("Low", "Mid", "High"),
    xlab = "Time (days)",
    ylab = "Survival probability",
    break.time.by = break_by,
    ggtheme = theme_classic(base_size = 12)
  )

  x_annot <- max_time * 0.25
  y_annot <- 0.13

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
      size = 3.2,
      hjust = 0,
      vjust = 0
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
      plot.margin = margin(8, 28, 8, 8)
    )

  ## Add n labels for available groups
  for (gg in c("Low", "Mid", "High")) {
    dd <- curve_end[as.character(curve_end$group) == gg, , drop = FALSE]
    if (nrow(dd) > 0) {
      g$plot <- g$plot +
        annotate(
          "text",
          x = dd$x_label[1],
          y = dd$y_label[1],
          label = dd$label[1],
          color = km_colors[gg],
          size = 3.4,
          hjust = 0,
          vjust = 0.5
        )
    }
  }

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

plot_nested_lrt <- function(km_hr_path, nested_lrt_path, file_path_main) {
    if (!file.exists(km_hr_path)) {
      stop(
        "KM-tertile HR matrix not found: ", km_hr_path,
        "\nRun Survival_analysis_KM_tertile_groups_smaller_censor_tumor_only.R first."
      )
    }

    HR_mat_km <- read.csv(
      km_hr_path,
      row.names = 1,
      check.names = FALSE
    )
    colnames(HR_mat_km) <- gsub("TCGA.", "TCGA-", colnames(HR_mat_km))

    ## ===============================
    ## Reload the nested LRT matrix exported above
    ## ===============================

    if (!file.exists(nested_lrt_path)) {
      stop(
        "Addmodules nested-LRT p-value matrix not found: ", nested_lrt_path,
        "\nThis file should be written earlier in this script."
      )
    }

    nestedLRT_pvalue_mat <- read.csv(
      nested_lrt_path,
      row.names = 1,
      check.names = FALSE
    )
    colnames(nestedLRT_pvalue_mat) <- gsub("TCGA.", "TCGA-", colnames(nestedLRT_pvalue_mat))

    ## ===============================
    ## Choose MI and cohorts
    ## ===============================
    mi_use <- "MI4"

    cohort_use <- c(
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

    highlight_cohorts <- c(
      "TCGA-BRCA",
      "TCGA-LIHC",
      "TCGA-LUAD",
      "TCGA-UCEC"
    )

    ## Safety checks
    missing_hr_cohorts <- setdiff(cohort_use, colnames(HR_mat_km))
    missing_lrt_cohorts <- setdiff(cohort_use, colnames(nestedLRT_pvalue_mat))
    if (length(missing_hr_cohorts) > 0) {
      stop("Missing cohorts in KM HR matrix: ", paste(missing_hr_cohorts, collapse = ", "))
    }
    if (length(missing_lrt_cohorts) > 0) {
      stop("Missing cohorts in nested-LRT matrix: ", paste(missing_lrt_cohorts, collapse = ", "))
    }
    if (!(mi_use %in% rownames(HR_mat_km))) {
      stop("MI not found in KM HR matrix: ", mi_use)
    }
    if (!(mi_use %in% rownames(nestedLRT_pvalue_mat))) {
      stop("MI not found in nested-LRT matrix: ", mi_use)
    }

    ## ===============================
    ## Build plotting dataframe
    ## x = HR from KM-tertile tumor-only script
    ## y = -log10(addmodules nested LRT p-value)
    ## ===============================
    df_scatter <- data.frame(
      Cohort = cohort_use,
      HR = as.numeric(HR_mat_km[mi_use, cohort_use]),
      pvalue = as.numeric(nestedLRT_pvalue_mat[mi_use, cohort_use]),
      stringsAsFactors = FALSE
    )

    df_scatter <- df_scatter[complete.cases(df_scatter), , drop = FALSE]

    if (nrow(df_scatter) == 0) {
      stop("No complete HR / nested-LRT p-value pairs available for scatter plot.")
    }

    df_scatter$neglog10_pvalue <- -log10(df_scatter$pvalue)
    df_scatter$Cohort_label <- gsub("TCGA-", "", df_scatter$Cohort)

    df_scatter$Group <- ifelse(
      df_scatter$Cohort %in% highlight_cohorts,
      "Significant worse survival",
      "Non-significant"
    )

    df_scatter$PointSize <- ifelse(
      df_scatter$Group == "Significant worse survival",
      4.8,
      3.8
    )

    print(df_scatter)

    scatter_table_path <- paste0(
      file_path_main,
      mi_use,
      "_KMscriptHR_vs_nestedLRT_scatterplot_table_addmodules_tumor_only.csv"
    )
    write.csv(df_scatter, scatter_table_path, row.names = FALSE)

    ## ===============================
    ## Scatter plot
    ## ===============================
    p_scatter <- ggplot(
      df_scatter,
      aes(x = HR, y = neglog10_pvalue)
    ) +
      geom_hline(
        yintercept = -log10(0.05),
        linetype = "dashed",
        linewidth = 0.5,
        color = "black"
      ) +
      geom_vline(
        xintercept = 1,
        linetype = "dashed",
        linewidth = 0.5,
        color = "black"
      ) +
      geom_point(
        aes(fill = Group, size = Group),
        shape = 21,
        color = "black",
        stroke = 0.4
      ) +
      ggrepel::geom_text_repel(
        aes(
          label = Cohort_label,
          fontface = ifelse(Group == "Significant worse survival", "bold", "plain")
        ),
        size = 3.6,
        box.padding = 0.3,
        point.padding = 0.25,
        segment.color = "grey50",
        max.overlaps = Inf,
        show.legend = FALSE
      ) +
      scale_fill_manual(
        values = c(
          "Significant worse survival" = "#E9242D",
          "Non-significant" = "#BDBDBD"
        ),
        breaks = c("Significant worse survival", "Non-significant")
      ) +
      scale_size_manual(
        values = c(
          "Significant worse survival" = 4.8,
          "Non-significant" = 3.8
        ),
        breaks = c("Significant worse survival", "Non-significant")
      ) +
      guides(
        size = "none",
        fill = guide_legend(
          override.aes = list(
            shape = 21,
            color = "black",
            size = c(4.8, 3.8)
          )
        )
      ) +
      theme_classic(base_size = 12) +
      theme(
        legend.title = element_blank(),
        legend.position = c(0.78, 0.22),
        legend.background = element_blank(),
        legend.key = element_blank(),
        plot.title = element_text(hjust = 0.5)
      ) +
      labs(
        x = "Hazard ratio from KM-tertile script",
        y = expression(-log[10]("nested LRT P-value")),
        title = paste0(mi_use, ": KM-script HR vs nested-model significance")
      )


    ggsave(
      filename = paste0(
        file_path_main,
        mi_use,
        "_KMscriptHR_vs_nestedLRT_scatterplot_addmodules_tumor_only_highlight4_redgray_legend.pdf"
      ),
      plot = p_scatter,
      width = 5.2,
      height = 4.2,
      device = cairo_pdf
    )
    invisible(list(HR = HR_mat_km, LRT_pvalue = nestedLRT_pvalue_mat,
                   scatter = df_scatter, output_dir = file_path_main))
}

plot_signed_heatmap <- function(mat, liptak_df, file_name, title, features_keep = NULL,
                                order_rows_by_mean_rank = FALSE,
                                features_drop = character(0),
                                color_limit = NULL,
                                icb_analysis_outdir, HEATMAP_TOP_N_FEATURES = 60,
                                HEATMAP_FILL_NA_WITH_ZERO = TRUE,
                                FORMAT_MI_LABELS_WITH_DASH = TRUE) {
  format_feature_label_for_plot <- function(x) {
    x <- as.character(x)
    if (!exists("FORMAT_MI_LABELS_WITH_DASH") || !isTRUE(FORMAT_MI_LABELS_WITH_DASH)) {
      return(x)
    }
    ## Format MI identifiers while preserving biomarker names such as MIAS_score.
    sub("^MI([0-9]+)$", "MI-\\1", x)
  }

  if (!requireNamespace("pheatmap", quietly = TRUE)) {
    warning("pheatmap not installed; skip heatmap: ", file_name)
    return(invisible(NULL))
  }
  if (nrow(mat) == 0 || ncol(mat) == 0) return(invisible(NULL))
  features_drop <- unique(as.character(features_drop))
  features_drop <- features_drop[!is.na(features_drop) & features_drop != ""]

  ## Select up to HEATMAP_TOP_N_FEATURES by absolute combined association,
  ## then rank this selected set within settings for the response heatmaps.
  if (is.null(features_keep)) {
    liptak_df <- liptak_df[is.finite(liptak_df$signed_log10p_Liptak), , drop = FALSE]
    if (length(features_drop) > 0) {
      liptak_df <- liptak_df[!(liptak_df$Feature %in% features_drop), , drop = FALSE]
    }
    liptak_df <- liptak_df[order(abs(liptak_df$signed_log10p_Liptak), decreasing = TRUE), , drop = FALSE]
    features_keep <- head(liptak_df$Feature, HEATMAP_TOP_N_FEATURES)
  }
  features_keep <- setdiff(features_keep, features_drop)
  features_keep <- intersect(features_keep, rownames(mat))
  if (length(features_keep) == 0) return(invisible(NULL))
  mat_show <- mat[features_keep, , drop = FALSE]

  if (isTRUE(order_rows_by_mean_rank)) {
    ## Rank signed association values separately within every cohort-treatment
    ## panel. Missing associations are not assigned a rank and do not
    ## contribute to that feature's mean rank.
    rank_mat <- vapply(
      seq_len(ncol(mat_show)),
      function(j) rank(mat_show[, j], ties.method = "average", na.last = "keep"),
      numeric(nrow(mat_show))
    )
    rank_mat <- matrix(
      rank_mat,
      nrow = nrow(mat_show),
      ncol = ncol(mat_show),
      dimnames = dimnames(mat_show)
    )
    mean_rank <- rowMeans(rank_mat, na.rm = TRUE)
    n_panels_ranked <- rowSums(is.finite(rank_mat))
    mean_rank[!is.finite(mean_rank)] <- Inf

    ## Primary ordering is mean rank. The raw row mean and feature name are
    ## deterministic tie-breakers only.
    raw_row_mean <- rowMeans(mat_show, na.rm = TRUE)
    raw_row_mean[!is.finite(raw_row_mean)] <- Inf
    row_order <- order(mean_rank, raw_row_mean, rownames(mat_show))
    mat_show <- mat_show[row_order, , drop = FALSE]
    rank_mat <- rank_mat[row_order, , drop = FALSE]

    mean_rank_out <- data.frame(
      Feature_raw = rownames(mat_show),
      Feature_display = format_feature_label_for_plot(rownames(mat_show)),
      mean_rank_across_cohort_treatment_panels = mean_rank[row_order],
      n_panels_ranked = n_panels_ranked[row_order],
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    panel_rank_out <- as.data.frame(rank_mat, check.names = FALSE)
    colnames(panel_rank_out) <- paste0("rank__", colnames(panel_rank_out))
    mean_rank_out <- cbind(mean_rank_out, panel_rank_out)
    write.csv(
      mean_rank_out,
      file.path(icb_analysis_outdir, paste0(tools::file_path_sans_ext(file_name), "_meanrank_order.csv")),
      row.names = FALSE
    )
  }

  ## Missing tests retain NA in result tables but display as neutral values.
  if (isTRUE(HEATMAP_FILL_NA_WITH_ZERO)) mat_show[is.na(mat_show)] <- 0

  ## Format heatmap row labels for display only. Raw feature names are preserved
  ## in all association result tables and are also saved as Feature_raw above.
  rownames(mat_show) <- make.unique(format_feature_label_for_plot(rownames(mat_show)))

  ## Fix color scale if color_limit is supplied. Values outside the color range
  ## are clipped only for plotting to avoid pheatmap rendering them as NA.
  if (!is.null(color_limit)) {
    color_limit <- as.numeric(color_limit)[1]
    if (!is.finite(color_limit) || color_limit <= 0) stop("color_limit must be a positive finite number.")
    mat_plot <- pmax(pmin(mat_show, color_limit), -color_limit)
    breaks <- seq(-color_limit, color_limit, length.out = 101)
  } else {
    mat_plot <- mat_show
    max_abs <- max(abs(mat_show), na.rm = TRUE)
    if (!is.finite(max_abs) || max_abs <= 0) max_abs <- 1
    breaks <- seq(-max_abs, max_abs, length.out = 101)
  }
  rdBu_cols <- RColorBrewer::brewer.pal(11, "RdBu")[2:10]
  cols <- grDevices::colorRampPalette(rdBu_cols)(100)

  grDevices::pdf(file.path(icb_analysis_outdir, file_name), width = max(6, 0.24 * ncol(mat_show) + 2), height = max(4, 0.17 * nrow(mat_show) + 2))
  pheatmap::pheatmap(
    mat_plot,
    color = cols,
    breaks = breaks,
    border_color = NA,
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    fontsize_row = 8,
    fontsize_col = 8,
    angle_col = 45,
    main = title
  )
  grDevices::dev.off()
  invisible(NULL)
}


export_icb_mi4_order <- function(icb_analysis_outdir, FOCUS_COHORT_TREATMENT_PAIRS) {
  suppressPackageStartupMessages(library(dplyr))
response_long_path <- file.path(
  icb_analysis_outdir,
  "ICB_response_signed_log10p_long.csv"
)

if (!file.exists(response_long_path)) {
  stop(
    "Cannot find response long table:\n",
    response_long_path,
    "\n\nPlease check whether RUN_STEP3_ASSOCIATION <- TRUE has finished successfully."
  )
}

## Restrict cached association tables to the same four settings.
response_long <- read.csv(response_long_path, check.names = FALSE)
response_long <- response_long[
  paste(response_long$Project, response_long$TreatmentICB, sep = " | ") %in%
    FOCUS_COHORT_TREATMENT_PAIRS$CohortTreatment,
  , drop = FALSE
]

cat("Loaded response long table:\n")
cat(response_long_path, "\n")
cat("Dimension:", nrow(response_long), "rows x", ncol(response_long), "columns\n\n")

## Accept either stored MI identifier convention.
mi_feature_use <- "MI4"

if (!mi_feature_use %in% response_long$Feature) {
  if ("MI-4" %in% response_long$Feature) {
    mi_feature_use <- "MI-4"
  } else {
    stop(
      "Cannot find MI4 or MI-4 in response_long$Feature.\n",
      "Available MI features are:\n",
      paste(sort(unique(response_long$Feature[grepl("^MI", response_long$Feature)])), collapse = ", ")
    )
  }
}

## Order cohort-treatment settings by increasing MI-4 signed association.
mi4_response_order_df <- response_long %>%
  filter(Feature == mi_feature_use) %>%
  mutate(
    CohortTreatment = paste(Project, TreatmentICB, sep = " | "),
    MI4_response_heatmap_value = signed_log10p,
    MI4_response_direction = case_when(
      is.na(MI4_response_heatmap_value) ~ NA_character_,
      MI4_response_heatmap_value > 0 ~ "Responder-enriched / superior / ICB-benefit",
      MI4_response_heatmap_value < 0 ~ "Non-responder-enriched / inferior / ICB-resistance",
      TRUE ~ "No directional association"
    )
  ) %>%
  arrange(MI4_response_heatmap_value) %>%
  transmute(
    Rank_low_to_high = row_number(),
    X_axis_label = CohortTreatment,
    CancerType = CancerType,
    Project = Project,
    TreatmentICB = TreatmentICB,
    Feature = Feature,
    N = N,
    N_Response = N_Response,
    N_No_response = N_No_response,
    MI4_response_heatmap_value = MI4_response_heatmap_value,
    MI4_response_direction = MI4_response_direction,
    P_value = p_value,
    Effect_median_Response_minus_NoResponse = effect_median_Response_minus_NoResponse
  )

## Report the order and corresponding association statistics.
cat("\n====================================================\n")
cat("MI4 response heatmap x-axis order: low -> high\n")
cat("Low/negative = non-responder-enriched / inferior / resistance\n")
cat("High/positive = responder-enriched / superior / benefit\n")
cat("====================================================\n\n")

print(mi4_response_order_df$X_axis_label)

cat("\n====================================================\n")
cat("Detailed dataframe\n")
cat("====================================================\n\n")

print(mi4_response_order_df, row.names = FALSE)

## Save the detailed table for downstream figure assembly.
mi4_order_outfile <- file.path(
  icb_analysis_outdir,
  "MI4_response_heatmap_xaxis_order_low_to_high_detail.csv"
)

write.csv(mi4_response_order_df, mi4_order_outfile, row.names = FALSE)

cat("\nSaved:\n")
cat(mi4_order_outfile, "\n")


  invisible(mi4_response_order_df)
}

pancancer_prepare_survival <- function(opt) {
  # Reuse the original MI-only cohort code on its saved clinical data. No GDC
  # query, expression processing, MI projection, or nested-model refit is needed.
  clinical_path <- file.path(opt$results, "clinical_data_list_MIdecomposition_tumor_only.Rdata")
  required <- c(clinical_path,
    file.path(pancancer_projection_cache(opt), paste0("MI_decomposition_", pancancer_cohorts(), ".csv")),
    file.path(opt$results, paste0("TCGA_MIs_", c("HR", "pvalue"), "_matrix_MIdecomposition_tumor_only.csv")))
  if (any(!file.exists(required))) stop("Missing cache preparation inputs: ", paste(required[!file.exists(required)], collapse = ", "))
  if (!requireNamespace("survival", quietly = TRUE)) stop("R package missing: survival")
  if (opt$mode == "check") return(invisible(TRUE))
  suppressPackageStartupMessages(library(survival))
  script <- file.path(opt$script_dir, "Survival_analysis_KM_tertile_groups_smaller_censor_tumor_only.R")
  source_text <- paste(readLines(script, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  section <- function(start, end) {
    a <- gregexpr(start, source_text, fixed = TRUE)[[1]]
    b <- gregexpr(end, source_text, fixed = TRUE)[[1]]
    if (length(a) != 1L || length(b) != 1L || a < 0 || b <= a) stop("Original survival source boundaries changed: ", start)
    substr(source_text, a, b - 1L)
  }
  # Evaluate function definitions only; never source the full download workflow.
  work <- new.env(parent = environment())
  for (expr in parse(script)) {
    if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
        is.call(expr[[3]]) && identical(expr[[3]][[1]], as.name("function"))) eval(expr, work)
  }
  load(clinical_path, envir = work)
  work$mi_decomposition_dir <- pancancer_projection_cache(opt)
  work$MI_decomposition_list <- list()
  work$MI_unique <- paste0("MI", 1:11)
  work$km_mi_to_plot <- work$MI_unique
  work$MI_source_type <- "MIdecomposition"
  # Preserve statistical code; defer figure rendering to the plot-only command.
  work$plot_km_nature <- function(...) invisible(NULL)
  staging <- tempfile("KM_prepare_")
  dir.create(staging)
  work$file_path_main <- staging
  clinical_block <- section("  vital <-", "  ## ===============================\n  ## Log2 CPM")
  alignment_block <- section("  mi_decomposition_path <-", "  # Compute the separate nested Cox analysis")
  model_block <- section("  ## ===============================\n  ## per-project result vectors", "  ## ===============================\n  ## Store cohort-specific Cox results")
  # Expression column names equal clinical row names at this point in the full
  # script. An empty matrix supplies those same names; no expression is used by
  # the preserved MIdecomposition score branch.
  cohort_code <- parse(text = paste0(
    "for (project_cur in pancancer_cohorts()) {\n",
    "coldata <- clinical_data_list[[project_cur]]\n",
    "stopifnot(!is.null(coldata), !anyDuplicated(rownames(coldata)))\n",
    "expr <- matrix(numeric(0), nrow=0, ncol=nrow(coldata), dimnames=list(NULL,rownames(coldata)))\n",
    "expr_cpm <- expr\n", clinical_block, alignment_block, model_block,
    "checks[[project_cur]] <- data.frame(Project=project_cur, MI=MI_unique, HR=HR_list, p=pvalue_list)\n}\n"))
  work$checks <- list()
  eval(cohort_code, work)
  checks <- do.call(rbind, work$checks)
  if (nrow(checks) != 99L) stop("The saved clinical inputs did not reproduce all 99 cohort/MI analyses.")
  read_baseline <- function(metric) as.matrix(read.csv(file.path(opt$results,
    paste0("TCGA_MIs_", metric, "_matrix_MIdecomposition_tumor_only.csv")), row.names=1, check.names=FALSE))
  old_hr <- read_baseline("HR"); old_p <- read_baseline("pvalue")
  checks$original_HR <- old_hr[cbind(checks$MI, checks$Project)]
  checks$original_p <- old_p[cbind(checks$MI, checks$Project)]
  validation <- file.path(opt$output, "validation")
  dir.create(validation, recursive=TRUE, showWarnings=FALSE)
  write.csv(checks, file.path(validation, "KM_cache_preparation_regression.csv"), row.names=FALSE)
  if (!isTRUE(all.equal(checks$HR, checks$original_HR, tolerance=1e-12)) ||
      !isTRUE(all.equal(checks$p, checks$original_p, tolerance=1e-12))) {
    stop("Recovered survival results differ from the saved analysis; no KM cache was published. Review the regression CSV.")
  }
  files <- list.files(file.path(staging, "KM_plot_inputs"), pattern="[.]rds$", full.names=TRUE)
  expected <- as.vector(outer(pancancer_cohorts(), work$MI_unique, function(a,b) paste0(a,"_",b,".rds")))
  if (!setequal(basename(files), expected)) stop("Incomplete KM input inventory; no cache was published.")
  destination <- file.path(opt$results, "KM_plot_inputs")
  dir.create(destination, recursive=TRUE, showWarnings=FALSE)
  if (!all(file.copy(files, destination, overwrite=TRUE))) stop("Could not publish all KM inputs.")
  writeLines(c("Prepared from the original survival cohort code and saved tumor-only clinical/MI data.",
    "All 99 adjusted Cox HR and P values agree with the existing results (tolerance 1e-12).",
    paste("Clinical source:", clinical_path), paste("Projection source:", work$mi_decomposition_dir)),
    file.path(destination, "provenance.txt"))
  message("Prepared 99 KM drawing inputs; existing cohort/MI Cox results verified. Cache: ", destination)
  invisible(TRUE)
}

pancancer_options <- function(kind, script_dir) {
  args <- commandArgs(trailingOnly = TRUE)
  value <- function(flag, default = NULL) {
    at <- which(args == flag)
    if (!length(at)) return(default)
    if (length(at) != 1L || at == length(args) || startsWith(args[at + 1L], "--")) stop("Expected one value after ", flag)
    args[at + 1L]
  }
  allowed <- c("--check", "--plot-only", "--prepare-plot-inputs", "--stage", "--cache-dir", "--mi", "--help")
  unknown <- args[startsWith(args, "--") & !args %in% allowed]
  if (length(unknown)) stop("Unknown options: ", paste(unknown, collapse = ", "))
  if ("--help" %in% args) {
    cat("Options: --check [--plot-only], --plot-only, --prepare-plot-inputs (survival), --stage NAME, --cache-dir PATH, --mi MI4\n")
    cat("Projection stages: all, tcga, pseudobulk. Survival plotting stages: all, km, nested-lrt.\n")
    quit(save = "no", status = 0L)
  }
  root <- Sys.getenv("SPIDERNET_ROOT", "D:/SpiderNet")
  result <- list(
    kind = kind, script_dir = script_dir,
    mode = if ("--check" %in% args) "check" else if ("--prepare-plot-inputs" %in% args) "prepare" else if ("--plot-only" %in% args) "plot" else "full",
    prepare = "--prepare-plot-inputs" %in% args,
    plot_only = "--plot-only" %in% args, stage = value("--stage", "all"),
    cache_dir = value("--cache-dir"), mi = strsplit(value("--mi", "MI2,MI3,MI9"), ",", fixed = TRUE)[[1]],
    results = Sys.getenv("SPIDERNET_PANCANCER_RESULTS", file.path(root, "Results/Pancancer/V1/SpiderNet_Result_dim11")),
    output = Sys.getenv("SPIDERNET_PANCANCER_OUTPUT", file.path(script_dir, "output")),
    gdc = Sys.getenv("SPIDERNET_TCGA_GDC", "E:/TCGA_GDC"),
    icb = Sys.getenv("SPIDERNET_ICB_DATA", "D:/CellFlowMap/Pancancer/Data/ICB-portal/ICB_data/ICB_data"),
    cancersea = Sys.getenv("SPIDERNET_CANCERSEA", file.path(Sys.getenv("SPIDERNET_PANCANCER_DATA", file.path(root, "Data/Pancancer")), "CancerSEA_marker")))
  stages <- switch(kind, projection = c("all", "tcga", "pseudobulk"), survival = c("all", "km", "nested-lrt"), icb = "all")
  if (result$prepare && (kind != "survival" || result$plot_only || result$stage != "all")) stop("--prepare-plot-inputs is a separate survival mode; do not combine it with --plot-only or --stage.")
  if (!result$stage %in% stages) stop("Unsupported ", kind, " stage: ", result$stage)
  if (kind == "survival" && result$stage != "all" && !result$plot_only) stop("Survival stage selection is available only with --plot-only; full execution preserves the joint KM/Cox workflow.")
  result$mi <- toupper(gsub("[^A-Za-z0-9]", "", result$mi))
  if (any(!grepl("^MI([1-9]|10|11)$", result$mi))) stop("--mi must contain comma-separated identifiers from MI1 to MI11.")
  result
}

pancancer_cohorts <- function() c("TCGA-BRCA", "TCGA-COAD", "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-SKCM", "TCGA-OV", "TCGA-PRAD", "TCGA-UCEC")
pancancer_existing <- function(paths, required = NULL) {
  usable <- dir.exists(paths)
  if (length(required)) usable <- usable & vapply(paths, function(path) all(file.exists(file.path(path, required))), logical(1))
  found <- paths[usable]
  if (length(found)) found[1] else paths[1]
}
pancancer_projection_cache <- function(opt) pancancer_existing(c(
  file.path(opt$output, "projection", "MI_decomposition_tumor_only"),
  file.path(opt$results, "MI_decomposition_tumor_only")), paste0("MI_decomposition_", pancancer_cohorts(), ".csv"))
pancancer_cache <- function(opt, stage) {
  if (!is.null(opt$cache_dir)) return(opt$cache_dir)
  switch(stage,
    tcga = pancancer_projection_cache(opt),
    pseudobulk = pancancer_existing(c(file.path(opt$output, "projection/MI_decomposition/S26b_MI4"), file.path(opt$results, "MI_decomposition/S26b_MI4")), "MI_hat_decomposed_joint_gene_LR.csv"),
    km = pancancer_existing(c(file.path(opt$output, "survival"), opt$results), "KM_plot_inputs"),
    `nested-lrt` = pancancer_existing(c(file.path(opt$output, "survival/S27b_nestedLRT"), file.path(opt$results, "S27b_nestedLRT"), opt$results), pancancer_nested_name()),
    icb = pancancer_existing(c(file.path(opt$output, "icb/ICB_analysis_joint_gene_LR_all_datasets"), file.path(opt$results, "ICB_analysis_joint_gene_LR_all_datasets")), "ICB_response_signed_log10p_matrix.csv"))
}
pancancer_nested_name <- function() "TCGA_MIs_nestedLRT_pvalue_add_score_matrix_MIdecomposition_tumor_only_addInvasionAngiogenesisEMTHypoxia_addTumorPurity.csv"
pancancer_hr_path <- function(opt, nested_cache) {
  paths <- c(file.path(dirname(nested_cache), "TCGA_MIs_HR_matrix_MIdecomposition_tumor_only.csv"),
             file.path(nested_cache, "TCGA_MIs_HR_matrix_MIdecomposition_tumor_only.csv"),
             file.path(opt$results, "TCGA_MIs_HR_matrix_MIdecomposition_tumor_only.csv"))
  found <- paths[file.exists(paths)]
  if (length(found)) found[1] else paths[1]
}

pancancer_check <- function(opt) {
  paths <- character(0); packages <- character(0)
  loadings <- paste0("loading_", c("intrinsic", "sender", "receiver", "LR"), "_use.csv")
  selected <- function(x) opt$stage %in% c("all", x)
  if (!opt$plot_only) {
    if (opt$kind %in% c("projection", "icb")) paths <- c(paths, file.path(opt$results, loadings))
    if (opt$kind == "projection") {
      packages <- c("osqp", "Matrix")
      if (selected("tcga")) packages <- c(packages, "TCGAbiolinks", "SummarizedExperiment", "edgeR", "pheatmap", "RColorBrewer", "ggplot2", "dplyr", "tidyr", "tibble")
      if (selected("pseudobulk")) paths <- c(paths, file.path(opt$results, c("geneexp_mean_df.csv", "MI_mean_df.csv")))
    }
    if (opt$kind == "survival") {
      packages <- c("TCGAbiolinks", "SummarizedExperiment", "dplyr", "ggplot2", "survival", "survminer", "edgeR", "ggrepel")
      paths <- c(paths, file.path(pancancer_projection_cache(opt), paste0("MI_decomposition_", pancancer_cohorts(), ".csv")),
                 file.path(opt$cancersea, paste0(c("Invasion", "Angiogenesis", "EMT", "Hypoxia"), ".txt")))
    }
    if (opt$kind == "icb") {
      packages <- c("osqp", "Matrix", "edgeR", "pheatmap", "RColorBrewer", "dplyr")
      for (study in c("Gide et al", "Hugo et al", "Jung et al")) {
        folder <- file.path(opt$icb, study); paths <- c(paths, folder)
        if (dir.exists(folder) && !length(list.files(folder, pattern = "[.]([Rr]data|[Rr][Dd][Ss])$", ignore.case = TRUE))) {
          paths <- c(paths, file.path(folder, paste0(study, "_ICB_data_collector.rds")))
        }
      }
    }
  } else {
    if (opt$kind == "projection") {
      if (selected("tcga")) {
        paths <- c(paths, file.path(pancancer_cache(opt, "tcga"), paste0("MI_decomposition_", pancancer_cohorts(), ".csv")))
        packages <- c(packages, "pheatmap", "RColorBrewer", "ggplot2", "dplyr", "tidyr", "tibble")
      }
      if (selected("pseudobulk")) paths <- c(paths, file.path(pancancer_cache(opt, "pseudobulk"), "MI_hat_decomposed_joint_gene_LR.csv"), file.path(opt$results, "MI_mean_df.csv"))
    }
    if (opt$kind == "survival") {
      if (selected("km")) {
        km_cache <- file.path(pancancer_cache(opt, "km"), "KM_plot_inputs")
        paths <- c(paths, km_cache)
        if (dir.exists(km_cache)) {
          km_files <- list.files(km_cache, pattern = "[.]rds$", full.names = FALSE)
          cached_mis <- sub("^.*_(MI[0-9]+)[.]rds$", "\\1", km_files)
          missing_mis <- setdiff(opt$mi, cached_mis)
          if (length(missing_mis)) paths <- c(paths, file.path(km_cache, paste0("<cohort>_", missing_mis, ".rds")))
        }
        packages <- c(packages, "survival", "survminer", "ggplot2")
      }
      if (selected("nested-lrt")) {
        cache <- pancancer_cache(opt, "nested-lrt")
        paths <- c(paths, pancancer_hr_path(opt, cache), file.path(cache, pancancer_nested_name()))
        packages <- c(packages, "ggplot2", "ggrepel")
      }
    }
    if (opt$kind == "icb") {
      paths <- c(paths, file.path(pancancer_cache(opt, "icb"), c("ICB_response_signed_log10p_matrix.csv", "ICB_OS_signed_log10p_matrix.csv", "ICB_response_Liptak_summary.csv", "ICB_OS_Liptak_summary.csv", "ICB_response_signed_log10p_long.csv")), file.path(opt$results, "loading_sender_use.csv"))
      packages <- c("pheatmap", "RColorBrewer", "dplyr")
    }
  }
  missing_files <- unique(paths[!file.exists(paths)])
  packages <- unique(packages)
  missing_packages <- packages[!vapply(packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
  cat("Analysis:", opt$kind, "| stage:", opt$stage, "|", if (opt$plot_only) "saved-result plotting" else "full analysis", "\n")
  cat("Spatial results:", opt$results, "\nOutput:", opt$output, "\n")
  if (length(missing_files)) cat("Missing inputs:\n", paste(missing_files, collapse = "\n"), "\n")
  if (length(missing_packages)) cat("Missing R packages:", paste(missing_packages, collapse = ", "), "\n")
  if (length(missing_files) || length(missing_packages)) stop("Preflight failed; no fit or download was started.", call. = FALSE)
  if (!opt$plot_only && (opt$kind == "survival" || (opt$kind == "projection" && selected("tcga")))) cat("Full execution queries/downloads GDC as before; its cache directory is", opt$gdc, "\n")
  if (opt$mode != "check") {
    graphics_dir <- file.path(opt$output, opt$kind)
    dir.create(graphics_dir, recursive = TRUE, showWarnings = FALSE)
    implicit_pdf <- file.path(normalizePath(graphics_dir, winslash = "/", mustWork = TRUE), "Rplots.pdf")
    options(device = function(...) grDevices::pdf(file = implicit_pdf, ...))
  }
  cat("Preflight passed.\n")
  invisible(TRUE)
}

pancancer_dispatch <- function(opt) {
  if (isTRUE(opt$prepare)) return(pancancer_prepare_survival(opt))
  pancancer_check(opt)
  if (opt$mode == "check") return(invisible(TRUE))
  selected <- function(x) opt$stage %in% c("all", x)
  create <- function(path) {dir.create(path, recursive = TRUE, showWarnings = FALSE); path}
  read_matrix <- function(path) as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
  if (opt$kind == "projection") {
    if (selected("tcga")) plot_tcga_summaries(pancancer_cache(opt, "tcga"), create(file.path(opt$output, "projection/MI_decomposition_tumor_only")), pancancer_cohorts())
    if (selected("pseudobulk")) {
      cache <- pancancer_cache(opt, "pseudobulk")
      message("Reading pseudo-bulk projection: ", cache)
      projected <- read_matrix(file.path(cache, "MI_hat_decomposed_joint_gene_LR.csv"))
      observed <- read.csv(file.path(opt$results, "MI_mean_df.csv"), row.names = 1, check.names = FALSE)
      if (!all(rownames(projected) %in% rownames(observed))) stop("Observed pseudo-bulk table does not contain all projected samples.")
      observed <- observed[rownames(projected), , drop = FALSE]
      observed <- align_observed_mi(projected, observed)
      plot_pseudobulk_recovery(projected, observed, create(file.path(opt$output, "projection/MI_decomposition/S26b_MI4")))
    }
  }
  if (opt$kind == "survival") {
    suppressPackageStartupMessages(library(ggplot2))
    if (selected("km")) {
      suppressPackageStartupMessages({library(survival); library(survminer)})
      cache <- file.path(pancancer_cache(opt, "km"), "KM_plot_inputs")
      files <- list.files(cache, pattern = "[.]rds$", full.names = TRUE)
      files <- files[sub("^.*_(MI[0-9]+)[.]rds$", "\\1", basename(files)) %in% opt$mi]
      if (!length(files)) stop("No cached KM plotting inputs for ", paste(opt$mi, collapse = ", "), ". Run the full survival analysis with the desired --mi selection first.")
      for (path in files) {
        args <- readRDS(path)
        args$outdir <- create(file.path(opt$output, "survival/KM_plots_tumor_only"))
        do.call(plot_km_nature, args)
      }
    }
    if (selected("nested-lrt")) {
      cache <- pancancer_cache(opt, "nested-lrt")
      plot_nested_lrt(pancancer_hr_path(opt, cache), file.path(cache, pancancer_nested_name()), paste0(create(file.path(opt$output, "survival/S27b_nestedLRT")), "/"))
    }
  }
  if (opt$kind == "icb") {
    cache <- pancancer_cache(opt, "icb")
    output <- create(file.path(opt$output, "icb/ICB_analysis_joint_gene_LR_all_datasets"))
    response_mat <- read_matrix(file.path(cache, "ICB_response_signed_log10p_matrix.csv"))
    os_mat <- read_matrix(file.path(cache, "ICB_OS_signed_log10p_matrix.csv"))
    response_liptak <- read.csv(file.path(cache, "ICB_response_Liptak_summary.csv"), check.names = FALSE)
    os_liptak <- read.csv(file.path(cache, "ICB_OS_Liptak_summary.csv"), check.names = FALSE)
    mi_names <- rownames(read.csv(file.path(opt$results, "loading_sender_use.csv"), row.names = 1, check.names = FALSE))
    plot_signed_heatmap(response_mat, response_liptak, "Heatmap_Response_MI_only.pdf", "ICB response association: MI modules", features_keep = mi_names[mi_names %in% rownames(response_mat)], order_rows_by_mean_rank = TRUE, icb_analysis_outdir = output)
    plot_signed_heatmap(response_mat, response_liptak, "Heatmap_Response_top_features.pdf", "ICB response association: top features", order_rows_by_mean_rank = TRUE, features_drop = "TMEscore", color_limit = 2, icb_analysis_outdir = output)
    plot_signed_heatmap(os_mat, os_liptak, "Heatmap_OS_MI_only.pdf", "OS/PFS association: MI modules", features_keep = mi_names[mi_names %in% rownames(os_mat)], icb_analysis_outdir = output)
    plot_signed_heatmap(os_mat, os_liptak, "Heatmap_OS_top_features.pdf", "OS/PFS association: top features", icb_analysis_outdir = output)
    # Preserve the exact cached association tables alongside regenerated plots.
    for (name in c("ICB_response_signed_log10p_matrix.csv", "ICB_OS_signed_log10p_matrix.csv", "ICB_response_Liptak_summary.csv", "ICB_OS_Liptak_summary.csv", "ICB_response_signed_log10p_long.csv")) {
      from <- file.path(cache, name); to <- file.path(output, name)
      if (normalizePath(from, winslash = "/", mustWork = FALSE) != normalizePath(to, winslash = "/", mustWork = FALSE)) file.copy(from, to, overwrite = TRUE)
    }
    focus <- data.frame(Project = c("Hugo et al", "Gide et al", "Jung et al", "Gide et al"), TreatmentICB = c("Anti-PD-1", "PD1", "anti-PD1/PDL1", "ipiPD1"), stringsAsFactors = FALSE)
    focus$CohortTreatment <- paste(focus$Project, focus$TreatmentICB, sep = " | ")
    export_icb_mi4_order(output, focus)
  }
  invisible(TRUE)
}
