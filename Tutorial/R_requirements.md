# R requirements for SpiderNet tutorials

Some SpiderNet tutorials include R or R Markdown (`.R` / `.Rmd`) analyses for downstream visualization, enrichment analysis, decomposition, and survival analysis. These R dependencies are **not** included in the Python `requirements.txt` files and should be installed separately in R.

This file summarizes the R packages detected from the tutorial scripts and provides installation commands.

---

## Required CRAN packages

Install these packages from CRAN:

```r
install.packages(c(
  "circlize",
  "CVXR",
  "dplyr",
  "ggplot2",
  "ggtext",
  "igraph",
  "knitr",
  "Matrix",
  "osqp",
  "pheatmap",
  "RColorBrewer",
  "readr",
  "reshape2",
  "scales",
  "stringr",
  "survival",
  "survminer",
  "svglite",
  "tibble",
  "tidyr",
  "tidyverse"
))
```

---

## Required Bioconductor packages

Install Bioconductor first if needed:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
```

Then install the Bioconductor packages used by the tutorials:

```r
BiocManager::install(c(
  "edgeR",
  "maftools",
  "SummarizedExperiment",
  "TCGAbiolinks"
))
```

---

## Optional package check

After installation, you can verify that all required R packages are available:

```r
cran_packages <- c(
  "circlize",
  "CVXR",
  "dplyr",
  "ggplot2",
  "ggtext",
  "igraph",
  "knitr",
  "Matrix",
  "osqp",
  "pheatmap",
  "RColorBrewer",
  "readr",
  "reshape2",
  "scales",
  "stringr",
  "survival",
  "survminer",
  "svglite",
  "tibble",
  "tidyr",
  "tidyverse"
)

bioc_packages <- c(
  "edgeR",
  "maftools",
  "SummarizedExperiment",
  "TCGAbiolinks"
)

all_packages <- c(cran_packages, bioc_packages)

missing_packages <- all_packages[
  !vapply(all_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]

if (length(missing_packages) == 0) {
  message("All R tutorial dependencies are installed.")
} else {
  message("Missing R packages:")
  print(missing_packages)
}
```

---

## Package usage by tutorial group

The following packages were detected from the tutorial R scripts and R Markdown files.

### AgingBrain tutorials

Used in:

- `Tutorial/AgingBrain/MIOI_visualization.Rmd`
- `Tutorial/AgingBrain/MI_GOenrichment_analysis.Rmd`
- `Tutorial/AgingBrain/MI_circleplot_visualization.Rmd`

Detected packages:

```text
circlize
dplyr
ggplot2
readr
scales
stringr
tidyr
```

### HGSOC tutorials

Used in:

- `Tutorial/HGSOC/MI_CAF_functionalstate_visualization.Rmd`

Detected packages:

```text
dplyr
ggplot2
knitr
tibble
```

### Pancancer tutorials

Used in:

- `Tutorial/Pancancer/MIEnrichment_analysis.Rmd`
- `Tutorial/Pancancer/MI_link_visualization.Rmd`
- `Tutorial/Pancancer/Survival_analysis.R`
- `Tutorial/Pancancer/TCGA_MI_decomposition.R`
- `Tutorial/Pancancer/TCGA_MI_decomposition_V2.R`
- `Tutorial/Pancancer/cascade_GO.R`
- `Tutorial/Pancancer/pancancer_mi_decomposition_benchmark.R`

Detected packages:

```text
CVXR
Matrix
RColorBrewer
SummarizedExperiment
TCGAbiolinks
circlize
dplyr
edgeR
ggplot2
ggtext
igraph
maftools
osqp
pheatmap
readr
reshape2
scales
stringr
survival
survminer
svglite
tibble
tidyr
tidyverse
```

### PerturbFISH tutorials

Used in:

- `Tutorial/PerturbFISH/PerturbFISH_GO_enrichment.Rmd`

Detected packages:

```text
dplyr
ggplot2
stringr
```

---

## Notes

- These R packages are required only for the R/R Markdown tutorials. They are not needed for installing or importing the core SpiderNet Python package.
- Python dependencies are listed separately in `requirements.txt`, `requirements-tutorial.txt`, `requirements-benchmark.txt`, and `requirements-full-freeze.txt`.
- Some R packages, especially `TCGAbiolinks`, may require additional system libraries depending on the operating system.
- If a tutorial uses local data paths such as `D:/SpiderNet/Data/...` or `D:/SpiderNet/Results/...`, update those paths to match your local directory structure before running the script.
