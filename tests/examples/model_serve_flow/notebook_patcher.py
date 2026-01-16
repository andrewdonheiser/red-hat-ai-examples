"""Utility to patch notebooks at test time for automated execution."""
import copy
import json
import re
from pathlib import Path
from typing import Any

import nbformat
from nbformat import NotebookNode


class NotebookPatcher:
    """Patches notebooks for automated test execution without modifying source files."""

    def __init__(self, notebook_path: Path):
        self.notebook_path = notebook_path
        self.notebook = nbformat.read(str(notebook_path), as_version=4)
        self._original = copy.deepcopy(self.notebook)

    def inject_parameters_cell(self, parameters: dict[str, Any]) -> "NotebookPatcher":
        """Inject a parameters cell at the top of the notebook.

        Papermill looks for cells tagged with 'parameters' to know where
        to inject values. We create one dynamically.
        """
        param_lines = ["# Parameters (injected by test)"]
        for key, value in parameters.items():
            if isinstance(value, str):
                param_lines.append(f'{key} = "{value}"')
            elif isinstance(value, (list, dict)):
                param_lines.append(f"{key} = {json.dumps(value)}")
            else:
                param_lines.append(f"{key} = {value}")

        param_cell = nbformat.v4.new_code_cell("\n".join(param_lines))
        param_cell.metadata["tags"] = ["parameters"]

        # Insert after first markdown cell (usually title/description)
        insert_idx = 1
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                insert_idx = i
                break

        self.notebook.cells.insert(insert_idx, param_cell)
        return self

    def replace_hardcoded_paths(
        self, replacements: dict[str, str]
    ) -> "NotebookPatcher":
        """Replace hardcoded path assignments with parameterized values.

        Args:
            replacements: Dict mapping variable names to parameter references
                e.g., {"base_model_path": "base_model_path",  # use injected param
                       "base_results_dir": "base_results_dir"}
        """
        for cell in self.notebook.cells:
            if cell.cell_type != "code":
                continue

            source = cell.source
            modified = False

            for var_name, param_name in replacements.items():
                # Match patterns like: var_name = "..." or var_name = f"..."
                patterns = [
                    rf'^({var_name}\s*=\s*)f?"[^"]*"',  # var = "value" or f"value"
                    rf"^({var_name}\s*=\s*)f?'[^']*'",  # var = 'value'
                    rf"^({var_name}\s*=\s*)[^#\n]+",  # var = expression
                ]

                for pattern in patterns:
                    if re.search(pattern, source, re.MULTILINE):
                        # Comment out original and add parameterized version
                        source = re.sub(
                            pattern,
                            f"# Original: \\g<0>\n{var_name} = {param_name}  # Injected by test",
                            source,
                            count=1,
                            flags=re.MULTILINE,
                        )
                        modified = True
                        break

            if modified:
                cell.source = source

        return self

    def skip_cells_matching(self, patterns: list[str]) -> "NotebookPatcher":
        """Skip cells containing specific patterns (e.g., pip install, manual instructions).

        Wraps matching cells in 'if False:' to skip execution.
        """
        for cell in self.notebook.cells:
            if cell.cell_type != "code":
                continue

            for pattern in patterns:
                if re.search(pattern, cell.source):
                    cell.source = (
                        "# Skipped by test automation\nif False:\n    "
                        + cell.source.replace("\n", "\n    ")
                    )
                    break

        return self

    def comment_out_cells_with_pattern(self, pattern: str) -> "NotebookPatcher":
        """Comment out entire cells matching a pattern."""
        for cell in self.notebook.cells:
            if cell.cell_type == "code" and re.search(pattern, cell.source):
                cell.source = (
                    "# Commented out by test automation\n# "
                    + cell.source.replace("\n", "\n# ")
                )
        return self

    def get_patched_notebook(self) -> NotebookNode:
        """Return the patched notebook."""
        return self.notebook

    def save_to(self, output_path: Path) -> Path:
        """Save patched notebook to a new location."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(self.notebook, str(output_path))
        return output_path

    def reset(self) -> "NotebookPatcher":
        """Reset to original notebook state."""
        self.notebook = copy.deepcopy(self._original)
        return self


# Notebook-specific patching configurations
NOTEBOOK_PATCHES = {
    "Base_Accuracy_Benchmarking.ipynb": {
        "parameters": {
            "model_name": "RedHatAI/Llama-3.1-8B-Instruct",
            "tasks": ["arc_easy"],  # Reduced for testing
        },
        "path_replacements": {
            "model_name": "model_name",
            "base_model_path": "base_model_path",
            "base_results_dir": "base_results_dir",
        },
        "skip_patterns": [r"!pip install"],
    },
    "Base_Performance_Benchmarking.ipynb": {
        "parameters": {
            "vllm_target": "http://127.0.0.1:8000",
        },
        "path_replacements": {
            "base_model_path": "base_model_path",
        },
        "skip_patterns": [r"!pip install", r"vllm serve", r"guidellm benchmark"],
    },
    "Model_Compression.ipynb": {
        "parameters": {
            "num_calibration_samples": 64,  # Reduced for testing
            "max_sequence_length": 512,
        },
        "path_replacements": {
            "base_model_path": "base_model_path",
            "compressed_model_path": "compressed_model_path",
        },
        "skip_patterns": [r"!pip install"],
    },
    "Compressed_Accuracy_Benchmarking.ipynb": {
        "parameters": {
            "tasks": ["arc_easy"],
        },
        "path_replacements": {
            "compressed_model_path": "compressed_model_path",
            "compressed_results_dir": "compressed_results_dir",
        },
        "skip_patterns": [r"!pip install"],
    },
    "Compressed_Performance_Benchmarking.ipynb": {
        "parameters": {
            "vllm_target": "http://127.0.0.1:8001",
        },
        "path_replacements": {
            "compressed_model_path": "compressed_model_path",
        },
        "skip_patterns": [r"!pip install", r"vllm serve", r"guidellm benchmark"],
    },
}
