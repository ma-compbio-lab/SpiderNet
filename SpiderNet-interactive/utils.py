"""
Utility functions for discovering and categorizing SpiderNet plots
"""
from pathlib import Path
import re
import config


def discover_datasets():
    """
    Scan project directory for all datasets with SpiderNet results.

    Returns:
        dict: Dataset name -> metadata dict
    """
    datasets = {}

    for search_dir in config.SEARCH_DIRS:
        if not search_dir.exists():
            continue

        # Search for SpiderNet result directories
        for results_dir in search_dir.rglob(config.RESULT_DIR_PATTERN):
            # Get parent directory structure to determine dataset name
            # Example: Interactivetool/HGSOC/Results/SpiderNet_Result_dim15
            parts = results_dir.relative_to(config.SPIDERNET_ROOT).parts

            # Find dataset name (usually the directory before "Results")
            dataset_name = None
            for i, part in enumerate(parts):
                if part == "Results" and i > 0:
                    dataset_name = parts[i-1]
                    break

            if not dataset_name:
                # Fallback: use first non-standard directory name
                dataset_name = parts[0] if parts else "Unknown"

            # Look for Code directory with notebooks
            dataset_root = results_dir.parent.parent if results_dir.parent.name == "Results" else results_dir.parent
            code_dir = dataset_root / "Code"
            notebooks = []
            if code_dir.exists():
                notebooks = [nb.name for nb in code_dir.glob("*.ipynb")]

            # Count plots
            plot_count = count_plots(results_dir)

            datasets[dataset_name] = {
                "path": dataset_root,
                "results_dir": results_dir,
                "notebooks": sorted(notebooks),
                "plot_count": plot_count,
            }

    return datasets


def count_plots(results_dir):
    """Count number of PNG plots in results directory"""
    return len(list(results_dir.rglob("*.png")))


def categorize_plots(results_dir):
    """
    Intelligently categorize plots by analysis type based on filename patterns.

    Args:
        results_dir: Path to SpiderNet_Result_dim* directory

    Returns:
        dict: category -> list of plot dicts
    """
    results_dir = Path(results_dir)

    categories = {
        "basic": [],
        "subtype": [],
        "cascade": []
    }

    # Find all PNG files
    for png_file in results_dir.rglob("*.png"):
        relative_path = png_file.relative_to(results_dir)
        filename = png_file.name

        # Create plot metadata
        plot = {
            "filename": filename,
            "title": format_plot_title(filename),
            "description": get_plot_description(filename),
            "path": str(relative_path),
            "full_path": str(png_file),
            "pdf_available": png_file.with_suffix('.pdf').exists(),
        }

        # Categorize based on filename patterns
        if is_basic_analysis_plot(filename, relative_path):
            categories["basic"].append(plot)
        elif is_cascade_analysis_plot(filename, relative_path):
            categories["cascade"].append(plot)
        else:
            # Default to subtype analysis
            categories["subtype"].append(plot)

    # Sort plots within each category
    for category in categories:
        categories[category] = sorted(categories[category], key=lambda p: p["filename"])

    return categories


def is_basic_analysis_plot(filename, relative_path):
    """Determine if plot belongs to basic analysis"""
    filename_lower = filename.lower()
    path_str = str(relative_path).lower()

    basic_patterns = [
        "correlation",
        "lr_loading",
        "avg_mi",
        "mean_mi",
        "mi_intensity",
    ]

    return any(pattern in filename_lower or pattern in path_str for pattern in basic_patterns)


def is_cascade_analysis_plot(filename, relative_path):
    """Determine if plot belongs to cascade analysis"""
    filename_lower = filename.lower()
    path_str = str(relative_path).lower()

    cascade_patterns = [
        "colocal",
        "triple",
        "spatial_",
        "lrstrength",
        "ligandexpression",
        "receptorexpression",
        "caf_monocyte",
        "functional_monocyte",
        "go_monocyte",
        "insitu_high_order",
        "go_enrichment",
    ]

    return any(pattern in filename_lower or pattern in path_str for pattern in cascade_patterns)


def get_plot_description(filename):
    """
    Get detailed description for a plot based on its filename.

    Args:
        filename: str, plot filename

    Returns:
        str: Description of what the plot shows
    """
    name_lower = filename.lower()

    # Basic Analysis Descriptions
    if "lr_pairs_correlation" in name_lower or "lr pairs correlation" in name_lower:
        return "Correlation heatmap showing relationships between ligand-receptor (LR) pairs based on their co-expression patterns. Hierarchical clustering groups LR pairs with similar expression correlations across cells."

    if "correlation" in name_lower and "mi" in name_lower:
        return "Correlation matrix showing relationships between different meta-interactions (MIs). Positive correlations indicate MIs that tend to co-occur across the tissue."

    if "lr_loading" in name_lower or "loading" in name_lower:
        return "Ligand-receptor (LR) pathway loading scores showing which LR pathways are enriched in each meta-interaction."

    if "avg_mi_cellclass_pair" in name_lower:
        return "Average meta-interaction strengths across different sender-receiver cell type pairs. Shows which cell-cell communications are mediated by each MI."

    if "avg_mi_heatmap" in name_lower:
        return "Heatmap showing average MI activity patterns across cell types and spatial contexts."

    # Subtype Analysis - Cluster Overview
    if "cluster" in name_lower and "malignant" in name_lower and "umap" not in name_lower:
        return "Malignant cell clustering assignments based on MI profiles. Each cluster represents a distinct subtype with unique communication patterns."

    # Subtype Analysis - UMAP Visualizations
    if "umap" in name_lower and "cellsubtype" in name_lower:
        if "mi_umap" in name_lower:
            return "UMAP projection of malignant cells using MI profiles. Cell subtypes identified by their meta-interaction communication patterns."
        elif "banksy" in name_lower:
            return "UMAP projection using Banksy spatial features. Shows how cells cluster based on their spatial neighborhood composition."
        elif "x_umap" in name_lower:
            return "UMAP projection using gene expression data. Shows transcriptional variation across malignant cells."

    if "malignant_c5" in name_lower and "umap" in name_lower:
        if "mi_umap" in name_lower:
            return "Highlight of malignant subtype C5 cells in MI-UMAP space. Shows spatial distribution of this specific subtype."
        elif "banksy" in name_lower:
            return "Highlight of malignant subtype C5 cells in Banksy-UMAP space based on neighborhood features."
        elif "x_umap" in name_lower:
            return "Highlight of malignant subtype C5 cells in gene expression UMAP space."

    # Neighboring Cell Proportions
    if "neighboring" in name_lower and "umap" in name_lower:
        cell_type = ""
        if "b.cell" in name_lower:
            cell_type = "B cells"
        elif "endothelial" in name_lower:
            cell_type = "endothelial cells"
        elif "fibroblast" in name_lower:
            cell_type = "fibroblasts"
        elif "monocyte" in name_lower:
            cell_type = "monocytes"
        elif "tnk.cell" in name_lower:
            cell_type = "T/NK cells"
        elif "mast.cell" in name_lower:
            cell_type = "mast cells"
        elif "malignant" in name_lower:
            cell_type = "malignant cells"

        if "mi_umap" in name_lower:
            return f"Proportion of neighboring {cell_type} overlaid on MI-UMAP. Shows how neighborhood composition correlates with MI-based subtypes."
        elif "banksy" in name_lower:
            return f"Proportion of neighboring {cell_type} overlaid on Banksy-UMAP. Spatial neighborhood features drive this embedding."
        elif "x_umap" in name_lower:
            return f"Proportion of neighboring {cell_type} overlaid on gene expression UMAP."

    # Clinical Metadata
    if "stage" in name_lower and "umap" in name_lower:
        return "Cancer stage information overlaid on UMAP. Shows whether MI-based subtypes correlate with disease stage."

    if "site" in name_lower and "umap" in name_lower:
        return "Tumor site (anatomical location) overlaid on UMAP. Reveals site-specific patterns in cell subtypes."

    if "outcome" in name_lower and "umap" in name_lower:
        return "Patient outcome information overlaid on UMAP. Links cell subtypes to clinical outcomes."

    if "patient" in name_lower and "umap" in name_lower:
        return "Patient identity overlaid on UMAP. Shows patient-specific vs. shared subtype patterns."

    if "treatment" in name_lower and "umap" in name_lower:
        return "Treatment status overlaid on UMAP. Indicates whether subtypes associate with treatment history."

    if "omentum" in name_lower or "stageiv" in name_lower:
        return "Clinical covariate proportions across MI-based subtypes. Bar plot showing enrichment of specific clinical features."

    # Functional States
    if "functionalstate" in name_lower and "boxplot" in name_lower:
        return "Tumor functional state scores (angiogenesis, hypoxia, inflammation, metastasis) across MI-based subtypes. Box plots show distribution per subtype."

    if any(term in name_lower for term in ["angiogenesis", "hypoxia", "inflammation", "metastasis"]) and "receiving" in name_lower:
        return "Association between tumor functional state and receiving meta-interaction strength. Wilcoxon test compares high vs. low MI activity."

    # CAF Analysis
    if "caf" in name_lower and "sending" in name_lower:
        return "Cancer-associated fibroblast (CAF) marker expression vs. sending MI strength. Shows how CAF signaling correlates with specific MIs."

    # In-Silico Perturbation
    if "perturbation" in name_lower or "insilico" in name_lower:
        return "Computational perturbation analysis. Predicted effect of modulating specific MI on downstream gene expression or cell states."

    # Cascade Analysis - Colocalization
    if "colocal" in name_lower:
        return "MI-MI colocalization network showing which meta-interactions significantly co-occur in spatial neighborhoods, indicating potential cascades."

    # Cascade Analysis - Triplets
    if "triple" in name_lower and "heatmap" in name_lower:
        return "Cell type triplet enrichment across MI cascades. Shows prevalence of cell1→cell2→cell3 communication paths."

    # Cascade Analysis - Spatial
    if "spatial" in name_lower and name_lower.startswith("spatial_"):
        return "In situ visualization of MI cascade in tissue. Shows spatial localization of specific cell type triplet with highlighted MI activities."

    # Cascade Analysis - LR/Gene Programs
    if "lrstrength" in name_lower or "ligandexpression" in name_lower or "receptorexpression" in name_lower:
        return "Ligand-receptor pair expression and activity in MI cascade. Shows which LR pairs mediate the cascade communication."

    if "functional_" in name_lower or "go_" in name_lower:
        return "Gene program and GO enrichment analysis for cascade target cells. Reveals biological processes activated by the MI cascade."

    # Default
    return "SpiderNet analysis visualization showing spatial cell-cell communication patterns and meta-interactions."


def format_plot_title(filename):
    """
    Convert filename to human-readable title.

    Args:
        filename: str, e.g., "UMAP_Malignant_cellsubtypes_X_umap.png"

    Returns:
        str: Human-readable title
    """
    # Remove extension
    name = filename.replace('.png', '').replace('.pdf', '')

    # Special case: Spatial plots
    if name.startswith("Spatial_"):
        return format_spatial_title(name)

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Format special terms
    replacements = {
        ' MI ': ' MI-',
        'MI-': 'MI-',
        'UMAP': 'UMAP:',
        ' vs ': ' vs ',
        'Receiving': 'Receiving',
        'Sending': 'Sending',
        'box wilcox': '(Wilcoxon)',
        'thr0p50': 'threshold=0.50',
        'thr0p60': 'threshold=0.60',
        ' X umap': '(X UMAP)',
        ' MI UMAP': '(MI UMAP)',
        ' Banksy umap': '(Banksy UMAP)',
        'Heatmap ': 'Heatmap: ',
        'Barplot ': 'Barplot: ',
        'avg': 'Average',
        'prop': 'Proportion',
        'celltype': 'Cell-Type',
        'Malignant C5': 'Malignant Subtype C5',
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Capitalize first letter of each major word
    # But preserve MI, UMAP, CAF, GO, LR, etc.
    words = name.split()
    formatted_words = []
    for word in words:
        if word.upper() in ['MI', 'UMAP', 'CAF', 'GO', 'LR', 'TNK', 'MHC']:
            formatted_words.append(word.upper())
        elif word.lower() in ['vs', 'and', 'or', 'the', 'of', 'by', 'to', 'in']:
            formatted_words.append(word.lower())
        elif word[0].isupper():
            formatted_words.append(word)
        else:
            formatted_words.append(word.capitalize())

    title = ' '.join(formatted_words)

    # Clean up extra spaces
    title = re.sub(r'\s+', ' ', title).strip()

    return title


def format_spatial_title(name):
    """Format spatial cascade plot titles"""
    # Example: Spatial_3_Monocyte-Fibroblast-Malignant_MI-12_MI-10
    parts = name.split('_')

    if len(parts) >= 4:
        sample_num = parts[1]
        cell_types = parts[2].replace('-', ' → ')
        mi_info = '_'.join(parts[3:])

        # Format MI cascade
        mi_parts = mi_info.split('_')
        mi_cascade = ' → '.join([p.replace('MI-', 'MI-') for p in mi_parts if 'MI' in p])

        return f"Spatial View #{sample_num}: {cell_types} ({mi_cascade})"

    # Fallback
    return name.replace('_', ' ')


def organize_plots_by_section(plots, category):
    """
    Organize plots into sections based on analysis category.

    Args:
        plots: list of plot dicts
        category: str, one of 'basic', 'subtype', 'cascade'

    Returns:
        dict: section_name -> list of plots
    """
    sections = {}

    if category == "basic":
        sections = {
            "LR Pairs Correlation": [],
            "MI Correlation": [],
            "LR Loading Pathway Enrichment": [],
            "Cell-Type Pair Enrichment": [],
            "Overall MI Patterns": [],
        }

        for plot in plots:
            name = plot["filename"].lower()
            if "lr_pairs_correlation" in name or "lr pairs correlation" in name:
                sections["LR Pairs Correlation"].append(plot)
            elif "mi_correlation" in name or "correlation" in name:
                sections["MI Correlation"].append(plot)
            elif "lr" in name or "loading" in name:
                sections["LR Loading Pathway Enrichment"].append(plot)
            elif "cellclass" in name or "pair" in name:
                sections["Cell-Type Pair Enrichment"].append(plot)
            else:
                sections["Overall MI Patterns"].append(plot)

    elif category == "subtype":
        sections = {
            "Cluster Overview": [],
            "MI-UMAP Visualization": [],
            "Banksy-UMAP Visualization": [],
            "Gene Expression UMAP": [],
            "Neighboring Cell Proportions": [],
            "Functional States": [],
            "CAF Marker Analysis": [],
            "Clinical Metadata": [],
            "In-Silico Perturbation": [],
        }

        for plot in plots:
            name = plot["filename"].lower()

            # Cluster overview (clusters, louvain, not UMAP)
            if ("cluster" in name or "louvain" in name) and "umap" not in name:
                sections["Cluster Overview"].append(plot)
            # MI-UMAP visualization
            elif "umap" in name and "mi_umap" in name and "neighboring" not in name:
                sections["MI-UMAP Visualization"].append(plot)
            # Banksy-UMAP visualization
            elif "umap" in name and "banksy" in name and "neighboring" not in name:
                sections["Banksy-UMAP Visualization"].append(plot)
            # Gene Expression UMAP (X_umap)
            elif "umap" in name and "x_umap" in name and "neighboring" not in name:
                sections["Gene Expression UMAP"].append(plot)
            # Neighboring cell proportions
            elif "neighboring" in name:
                sections["Neighboring Cell Proportions"].append(plot)
            # Functional states
            elif any(term in name for term in ["angiogenesis", "hypoxia", "inflammation", "metastasis", "functionalstate"]):
                sections["Functional States"].append(plot)
            # CAF markers
            elif "caf" in name and "sending" in name:
                sections["CAF Marker Analysis"].append(plot)
            # Clinical metadata
            elif any(term in name for term in ["stage", "site", "omentum", "outcome", "patient", "treatment"]):
                sections["Clinical Metadata"].append(plot)
            # In-silico perturbation
            elif "perturbation" in name:
                sections["In-Silico Perturbation"].append(plot)

    elif category == "cascade":
        sections = {
            "MI-MI Colocalization": [],
            "Cell-Type Triplet Analysis": [],
            "Spatial Visualizations": [],
            "LR Pathway Analysis": [],
            "Gene Program Analysis": [],
            "Other": [],
        }

        for plot in plots:
            name = plot["filename"].lower()
            path = plot["path"].lower()

            if "colocal" in name:
                sections["MI-MI Colocalization"].append(plot)
            elif "triple" in name:
                sections["Cell-Type Triplet Analysis"].append(plot)
            elif "spatial" in name or "insitu" in path:
                sections["Spatial Visualizations"].append(plot)
            elif any(term in name for term in ["lrstrength", "ligandexpression", "receptorexpression"]):
                sections["LR Pathway Analysis"].append(plot)
            elif any(term in name for term in ["caf_", "functional_", "go_"]) or "go_enrichment" in path:
                sections["Gene Program Analysis"].append(plot)
            else:
                sections["Other"].append(plot)

    # Remove empty sections
    sections = {k: v for k, v in sections.items() if v}

    return sections
