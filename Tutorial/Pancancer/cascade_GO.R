suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(scales)
  library(stringr)
  library(svglite)
})

# ======================================================
# 1. Read data
# ======================================================
plot_df <- read_csv(
  "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/GO_summary_for_R/plot_df_long.csv",
  show_col_types = FALSE
)

CAF_deg <- read_csv(
  "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/CAFscore_pathway2_BothPos_vs_OnlyMIsecond_by_cancertype_Fibroblast-cancercell-cancercell.csv",
  show_col_types = FALSE
)

downstream_pathway_deg <- read_csv(
  "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/LRStrength_pathway2_BothPos_vs_OnlyMIsecond_by_cancertype_Fibroblast-cancercell-cancercell.csv",
  show_col_types = FALSE
)

# ======================================================
# 2. Representative GO terms by functional class
# ======================================================
GO_interferon_cytokine_inflammatory <- c(
  # "cytokine-mediated signaling pathway (GO:0019221)",
  "cellular response to cytokine stimulus (GO:0071345)"
  # "interferon-gamma-mediated signaling pathway (GO:0060333)"
)

GO_proliferation_cell_cycle_DNA_damage <- c(
  "positive regulation of cell population proliferation (GO:0008284)"
  # "mitotic G1 DNA damage checkpoint signaling (GO:0031571)",
  # "DNA damage response, signal transduction by p53 class mediator (GO:0030330)"
)

GO_kinase_PI3K_AKT_MAPK_signaling <- c(
  # "positive regulation of intracellular signal transduction (GO:1902533)",
  # "regulation of protein kinase B signaling (GO:0051896)",
  "MAPK cascade (GO:0000165)"
)

GO_ECM_adhesion_EMT_remodeling <- c(
  "extracellular matrix organization (GO:0030198)"
  # "extracellular structure organization (GO:0043062)"
  # "external encapsulating structure organization (GO:0045229)"
)

representative_GO_by_class <- list(
  `Interferon / cytokine / inflammatory signaling` = GO_interferon_cytokine_inflammatory,
  `Proliferation / cell cycle / DNA damage` = GO_proliferation_cell_cycle_DNA_damage,
  `Kinase / PI3K-AKT / MAPK signaling` = GO_kinase_PI3K_AKT_MAPK_signaling,
  `ECM remodeling / adhesion / EMT-like remodeling` = GO_ECM_adhesion_EMT_remodeling
)

# ======================================================
# 3. GO -> class mapping
# ======================================================
go_class_df <- enframe(representative_GO_by_class, name = "Feature_class") %>%
  unnest(value) %>%
  rename(GO_term = value)

# ======================================================
# 4. Helper functions
# ======================================================
safe_neglog10 <- function(x, cap = 6) {
  out <- suppressWarnings(-log10(x))
  out[is.infinite(out)] <- cap
  out <- pmin(out, cap)
  out
}

strip_go_id <- function(x) {
  gsub("\\s*\\(GO:\\d+\\)", "", x)
}

format_cancer_label <- function(x) {
  out <- stringr::str_extract(x, "(?<=Human).*?(?=Patient)")
  out[is.na(out) | out == ""] <- x[is.na(out) | out == ""]
  out
}

# ======================================================
# 5. GO program data
# ======================================================
plot_df_rep <- plot_df %>%
  inner_join(go_class_df, by = "GO_term") %>%
  transmute(
    CancerType,
    Feature       = strip_go_id(as.character(GO_term)),
    Feature_class = Feature_class,
    neglog10_pval = pmin(as.numeric(neglog10_pval), 6),
    size_raw      = as.numeric(GeneRatio),
    size_type     = "GeneRatio"
  ) %>%
  filter(!is.na(CancerType), !is.na(Feature), !is.na(neglog10_pval), !is.na(size_raw))

# ======================================================
# 6. CAF and MI2 downstream blocks
# ======================================================
CAF_df <- CAF_deg %>%
  transmute(
    CancerType,
    Feature       = "CAF score",
    Feature_class = "CAF activity",
    neglog10_pval = safe_neglog10(pval_adj, cap = 6),
    size_raw      = abs(as.numeric(logFC)),
    size_type     = "logFC"
  ) %>%
  filter(!is.na(CancerType), !is.na(neglog10_pval), !is.na(size_raw))

Pathway_df <- downstream_pathway_deg %>%
  transmute(
    CancerType,
    Feature       = "MI2 downstream activity",
    Feature_class = "MI2 downstream activity",
    neglog10_pval = safe_neglog10(pval_adj, cap = 6),
    size_raw      = abs(as.numeric(logFC)),
    size_type     = "logFC"
  ) %>%
  filter(!is.na(CancerType), !is.na(neglog10_pval), !is.na(size_raw))

# ======================================================
# 7. Merge all data
# ======================================================
plot_df_all <- bind_rows(CAF_df, plot_df_rep)

# ======================================================
# 8. Feature and block order
# ======================================================
GO_feature_order <- unlist(representative_GO_by_class, use.names = FALSE) %>%
  strip_go_id()

feature_order <- c(
  "CAF score",
  # "MI2 downstream activity",
  GO_feature_order
)

feature_class_order <- c(
  "CAF activity",
  # "MI2 downstream activity",
  names(representative_GO_by_class)
)

plot_df_all <- plot_df_all %>%
  mutate(
    Feature = factor(Feature, levels = feature_order),
    Feature_class = factor(Feature_class, levels = feature_class_order),
    Point_shape_group = case_when(
      Feature_class == "CAF activity" ~ "CAF activity",
      # Feature_class == "MI2 downstream activity" ~ "MI2 downstream activity",
      TRUE ~ "GO program"
    )
  )

# ======================================================
# 9. Shared size scaling
# ======================================================
plot_df_all <- plot_df_all %>%
  mutate(
    size_val = pmin(size_raw / 0.5, 1)
  )

# ======================================================
# 10. Use a fixed cancer type order
# ======================================================
cancer_order <- c(
  "UterineCancer",
  "ColonCancer",
  "LungCancer",
  "BreastCancer",
  "OvarianCancer",
  "MelanomaCancer",
  "LiverCancer",
  "ProstateCancer"
)
cancer_order <- rev(cancer_order)

plot_df_all <- plot_df_all %>%
  mutate(CancerType = factor(CancerType, levels = cancer_order))

# ======================================================
# 11. Final plotting data
# ======================================================
sig_cutoff <- -log10(0.05)

plot_df_plot <- plot_df_all %>%
  filter(!is.na(Feature), !is.na(Feature_class), !is.na(CancerType)) %>%
  filter(neglog10_pval > sig_cutoff)

# ======================================================
# 12. Plot
# ======================================================
p <- ggplot(plot_df_plot, aes(x = Feature, y = CancerType)) +
  geom_point(
    aes(
      size  = size_val,
      color = neglog10_pval,
      shape = Point_shape_group
    ),
    stroke = 0.3,
    na.rm  = TRUE
  ) +
  scale_shape_manual(
    values = c(
      "CAF activity" = 15,
      # "MI2 downstream activity" = 15,
      "GO program" = 16
    ),
    guide = "none"
  ) +
  scale_color_gradientn(
    colors = c("#1400FC", "#CA0089", "#FE0006"),
    limits = c(0, 6),
    oob    = scales::squish,
    name   = expression(-log[10]~"(adjusted P)")
  ) +
  scale_size_continuous(
    name   = "Effect size",
    range  = c(3, 11),
    limits = c(0, 1),
    breaks = c(0.25, 0.5, 0.75, 1),
    labels = c("0.125", "0.25", "0.375", "0.5")
  ) +
  facet_grid(
    ~ Feature_class,
    scales = "free_x",
    space  = "free_x",
    drop   = TRUE
  ) +
  scale_y_discrete(
    labels = function(x) format_cancer_label(x),
    drop = FALSE
  ) +
  labs(x = NULL, y = "Cancer type") +
  theme_classic(base_size = 8, base_family = "Arial") +
  theme(
    axis.line         = element_line(linewidth = 0.35, colour = "black"),
    axis.ticks        = element_line(linewidth = 0.35, colour = "black"),
    axis.ticks.length = unit(1.4, "mm"),
    
    strip.background  = element_blank(),
    strip.text.x      = element_text(face = "bold", size = 8, colour = "black"),
    
    axis.text.x       = element_text(
      angle = 60,
      hjust = 1,
      vjust = 1,
      size = 7,
      colour = "black"
    ),
    axis.text.y       = element_text(size = 8, colour = "black"),
    axis.title.y      = element_text(size = 9, colour = "black"),
    
    panel.spacing.x   = unit(2.5, "mm"),
    plot.margin       = margin(4, 6, 4, 6, unit = "mm"),
    
    legend.position   = "bottom",
    legend.title      = element_text(size = 8),
    legend.text       = element_text(size = 7),
    legend.key.height = unit(3.0, "mm"),
    legend.key.width  = unit(10, "mm")
  )

print(p)

# ======================================================
# 13. Save
# ======================================================
output_prefix <- "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/GO_MI2_CAF_singlepanel_final"

ggsave(
  filename = paste0(output_prefix, ".pdf"),
  plot     = p,
  width    = 8.5,
  height   = 8.7,
  units    = "in",
  device   = cairo_pdf
)

ggsave(
  filename = paste0(output_prefix, ".png"),
  plot     = p,
  width    = 8.5,
  height   = 8.7,
  units    = "in",
  dpi      = 600
)