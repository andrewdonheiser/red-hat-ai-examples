"""KFP component for base model accuracy benchmarking."""
from kfp import dsl
from kfp.dsl import Dataset, Output

# Base image with GPU support for RHOAI
BASE_IMAGE = "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/pytorch:2024.2-py312-cuda"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "papermill",
        "nbformat",
        "ipykernel",
        "torch",
        "transformers",
        "lm-eval",
        "accelerate",
    ],
)
def base_accuracy_benchmarking_component(
    model_name: str,
    tasks: str,  # JSON array string e.g. '["mmlu", "arc_easy"]'
    base_model_output: Output[Dataset],
    accuracy_results_output: Output[Dataset],
    limit: int = None,
    batch_size: str = "auto",
):
    """Execute base accuracy benchmarking notebook via papermill with test-time patching.

    Args:
        model_name: HuggingFace model name (e.g., 'RedHatAI/Llama-3.1-8B-Instruct')
        tasks: JSON array of evaluation tasks
        base_model_output: Output artifact for saved base model
        accuracy_results_output: Output artifact for accuracy results
        limit: Limit number of samples per task (None for full eval)
        batch_size: Batch size for evaluation
    """
    import json
    import sys
    import papermill as pm
    from pathlib import Path

    # Add kfp_components to path for NotebookPatcher import
    sys.path.insert(0, "/opt/app-root/src/tests/examples/model_serve_flow")
    from notebook_patcher import NotebookPatcher

    # Set up paths
    notebook_dir = Path(
        "/opt/app-root/src/examples/model-serve-flow/01_Base_Accuracy_Benchmarking"
    )
    notebook_path = notebook_dir / "Base_Accuracy_Benchmarking.ipynb"

    base_model_path = Path(base_model_output.path) / model_name.replace("/", "-")
    results_dir = Path(accuracy_results_output.path)

    base_model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Patch notebook
    test_params = {
        "model_name": model_name,
        "base_model_path": str(base_model_path),
        "base_results_dir": str(results_dir),
        "tasks": tasks,
    }

    patcher = NotebookPatcher(notebook_path)
    patcher.inject_parameters_cell(test_params)
    patcher.replace_hardcoded_paths(
        {
            "model_name": "model_name",
            "base_model_path": "base_model_path",
            "base_results_dir": "base_results_dir",
            "tasks": "tasks",
        }
    )
    patcher.skip_cells_matching([r"!pip install"])

    patched_path = Path("/tmp/patched_base_accuracy.ipynb")
    patcher.save_to(patched_path)

    # Execute
    output_notebook = Path("/tmp/executed_base_accuracy.ipynb")
    pm.execute_notebook(
        str(patched_path),
        str(output_notebook),
        cwd=str(notebook_dir),
        kernel_name="python3",
    )

    print(f"Base model saved to: {base_model_path}")
    print(f"Accuracy results saved to: {results_dir}")
