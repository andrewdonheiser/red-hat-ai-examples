"""KFP component for model compression."""
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
        "llmcompressor",
        "datasets",
        "accelerate",
    ],
)
def model_compression_component(
    base_model_input: Input[Dataset],
    compressed_model_output: Output[Dataset],
    quantization_scheme: str = "W8A8",
    num_calibration_samples: int = 512,
    max_sequence_length: int = 1024,
    smoothing_strength: float = 0.8,
):
    """Compress base model using LLM Compressor with test-time notebook patching."""
    import sys
    import papermill as pm
    from pathlib import Path

    # Add kfp_components to path for NotebookPatcher import
    sys.path.insert(0, "/opt/app-root/src/tests/examples/model_serve_flow")
    from notebook_patcher import NotebookPatcher

    base_model_path = Path(base_model_input.path)
    compressed_model_path = Path(compressed_model_output.path)
    compressed_model_path.mkdir(parents=True, exist_ok=True)

    # Find actual model directory
    model_dirs = list(base_model_path.glob("*"))
    actual_base_path = model_dirs[0] if model_dirs else base_model_path

    compressed_subdir = compressed_model_path / f"{actual_base_path.name}-int8-dynamic"

    notebook_dir = Path(
        "/opt/app-root/src/examples/model-serve-flow/03_Model_Compression"
    )
    notebook_path = notebook_dir / "Model_Compression.ipynb"

    test_params = {
        "base_model_path": str(actual_base_path),
        "compressed_model_path": str(compressed_subdir),
        "num_calibration_samples": num_calibration_samples,
        "max_sequence_length": max_sequence_length,
        "scheme": quantization_scheme,
        "smoothing_strength": smoothing_strength,
    }

    patcher = NotebookPatcher(notebook_path)
    patcher.inject_parameters_cell(test_params)
    patcher.replace_hardcoded_paths(
        {
            "base_model_path": "base_model_path",
            "compressed_model_path": "compressed_model_path",
        }
    )
    patcher.skip_cells_matching([r"!pip install"])

    patched_path = Path("/tmp/patched_compression.ipynb")
    patcher.save_to(patched_path)

    pm.execute_notebook(
        str(patched_path),
        str(Path("/tmp/executed_compression.ipynb")),
        cwd=str(notebook_dir),
        kernel_name="python3",
    )

    print(f"Compressed model saved to: {compressed_subdir}")
