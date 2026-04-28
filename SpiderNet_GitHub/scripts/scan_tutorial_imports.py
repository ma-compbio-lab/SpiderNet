from pathlib import Path
import ast
import json
import re
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_ROOT = PROJECT_ROOT / "Tutorial"

stdlib = {
    "os", "sys", "pathlib", "typing", "time", "json", "pickle",
    "subprocess", "warnings", "glob", "collections", "itertools",
    "math", "random", "copy", "re", "shutil", "functools",
    "datetime", "dataclasses", "multiprocessing", "logging",
    "argparse", "gc", "__future__",
}

internal = {
    "SpiderNet",
    "simulation_benchmark_utils",
}

module_to_pip = {
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "torch_geometric": "torch-geometric",
    "torch_scatter": "torch-scatter",
    "torch_sparse": "torch-sparse",
}

imports = defaultdict(set)


def add_import(module_name, file_path):
    if not module_name:
        return
    top_level = module_name.split(".")[0]
    if top_level not in stdlib and top_level not in internal:
        imports[top_level].add(str(file_path.relative_to(PROJECT_ROOT)))


def scan_code(code, file_path):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        pattern = re.compile(
            r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_\.]*)",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            add_import(match.group(1), file_path)
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_import(alias.name, file_path)
        elif isinstance(node, ast.ImportFrom):
            add_import(node.module, file_path)


for path in TUTORIAL_ROOT.rglob("*"):
    if path.suffix == ".py":
        code = path.read_text(encoding="utf-8", errors="ignore")
        scan_code(code, path)

    elif path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        code_cells = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                code_cells.append(source)
        scan_code("\n".join(code_cells), path)

    elif path.suffix in {".Rmd", ".R"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"library\(([^)]+)\)", text):
            add_import("R::" + match.group(1).strip().strip('"').strip("'"), path)
        for match in re.finditer(r"require\(([^)]+)\)", text):
            add_import("R::" + match.group(1).strip().strip('"').strip("'"), path)


print("\nDetected non-stdlib imports:\n")
for module in sorted(imports):
    pip_name = module_to_pip.get(module, module)
    print(f"{module:30s} -> {pip_name}")

print("\nDetailed usage:\n")
for module in sorted(imports):
    print(f"\n[{module}]")
    for file_path in sorted(imports[module]):
        print(f"  - {file_path}")