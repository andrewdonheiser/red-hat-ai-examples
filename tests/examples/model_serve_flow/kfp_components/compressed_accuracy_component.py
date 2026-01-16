"""KFP component for compressed model accuracy benchmarking."""
from kfp import dsl
from kfp.dsl import Dataset, Output, Input

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
def compressed_accuracy_benchmarking_component(
    compressed_model_input: Input[Dataset],
    accuracy_results_output: Output[Dataset],
    tasks: str,  # JSON array string
    limit: int = None,
    batch_size: int = 16,
):
    """Execute compressed model accuracy benchmarking with test-time patching."""
    import sys
    import papermill as pm
    from pathlib import Path

    # Add kfp_components to path for NotebookPatcher import
    sys.path.insert(0, "/opt/app-root/src/tests/examples/model_serve_flow")
    from notebook_patcher import NotebookPatcher

    compressed_model_path = Path(compressed_model_input.path)
    results_dir = Path(accuracy_results_output.path)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find actual model directory
    model_dirs = list(compressed_model_path.glob("*"))
    actual_model_path = model_dirs[0] if model_dirs else compressed_model_path

    notebook_dir = Path(
        "/opt/app-root/src/examples/model-serve-flow/04_Compressed_Accuracy_Benchmarking"
    )
    notebook_path = notebook_dir / "Compressed_Accuracy_Benchmarking.ipynb"

    test_params = {
        "compressed_model_path": str(actual_model_path),
        "compressed_results_dir": str(results_dir),
        "tasks": tasks,
    }

    patcher = NotebookPatcher(notebook_path)
    patcher.inject_parameters_cell(test_params)
    patcher.replace_hardcoded_paths(
        {
            "compressed_model_path": "compressed_model_path",
            "compressed_results_dir": "compressed_results_dir",
            "tasks": "tasks",
        }
    )
    patcher.skip_cells_matching([r"!pip install"])

    patched_path = Path("/tmp/patched_compressed_accuracy.ipynb")
    patcher.save_to(patched_path)

    pm.execute_notebook(
        str(patched_path),
        str(Path("/tmp/executed_compressed_accuracy.ipynb")),
        cwd=str(notebook_dir),
        kernel_name="python3",
    )

    print(f"Compressed accuracy results saved to: {results_dir}")
