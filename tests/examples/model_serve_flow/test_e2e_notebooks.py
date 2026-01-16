"""End-to-end tests for model-serve-flow notebooks using test-time patching.

Tests can be run individually or together. When run together, pytest-dependency
ensures correct execution order. When run individually, tests check for required
artifacts and skip with a clear message if dependencies are missing.

Dependency graph:
    test_base_accuracy_full
        ├── test_base_performance_benchmark (needs base_model)
        └── test_model_compression (needs base_model)
                ├── test_compressed_accuracy (needs compressed_model)
                └── test_compressed_performance_benchmark (needs compressed_model)
"""
import os
from pathlib import Path

import papermill as pm
import pytest

from .notebook_patcher import NotebookPatcher, NOTEBOOK_PATCHES


def _get_model_paths(base_model_dir, compressed_model_dir):
    """Get model paths based on environment configuration."""
    model_name = os.environ.get(
        "TEST_MODEL_NAME", "RedHatAI/Llama-3.1-8B-Instruct"
    )
    model_subdir = model_name.replace("/", "-")
    base_path = base_model_dir / model_subdir
    compressed_path = compressed_model_dir / f"{model_subdir}-int8-dynamic"
    return model_name, base_path, compressed_path


def _skip_if_missing(path: Path, artifact_name: str, dependency_test: str):
    """Skip test with clear message if required artifact is missing."""
    if not path.exists():
        pytest.skip(
            f"{artifact_name} not found at {path}. "
            f"Run {dependency_test} first, or run all tests together."
        )


class TestBaseAccuracyBenchmarking:
    """E2E tests for 01_Base_Accuracy_Benchmarking notebook."""

    NOTEBOOK_DIR = "01_Base_Accuracy_Benchmarking"
    NOTEBOOK_NAME = "Base_Accuracy_Benchmarking.ipynb"

    @pytest.mark.e2e
    @pytest.mark.gpu
    @pytest.mark.timeout(14400)  # 4 hours
    @pytest.mark.dependency(name="base_accuracy")
    def test_base_accuracy_full(
        self,
        model_serve_flow_path,
        base_model_dir,
        results_dir,
        executed_notebooks_dir,
        notebook_patcher,
    ):
        """Run base accuracy benchmarking with patched notebook."""
        notebook_path = model_serve_flow_path / self.NOTEBOOK_DIR / self.NOTEBOOK_NAME

        # Get model name from env or use default
        model_name = os.environ.get(
            "TEST_MODEL_NAME", "RedHatAI/Llama-3.1-8B-Instruct"
        )
        model_subdir = model_name.replace("/", "-")

        # Define test parameters
        test_params = {
            "model_name": model_name,
            "base_model_path": str(base_model_dir / model_subdir),
            "base_results_dir": str(results_dir / "base_accuracy"),
            "tasks": ["arc_easy"],  # Reduced for testing
        }

        # Patch notebook
        patcher = notebook_patcher(notebook_path)
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

        # Save patched notebook
        patched_path = executed_notebooks_dir / f"patched_{self.NOTEBOOK_NAME}"
        patcher.save_to(patched_path)

        # Execute
        output_path = executed_notebooks_dir / f"executed_{self.NOTEBOOK_NAME}"
        pm.execute_notebook(
            str(patched_path),
            str(output_path),
            cwd=str(notebook_path.parent),
            kernel_name="python3",
        )

        # Verify outputs
        assert Path(test_params["base_model_path"]).exists(), "Base model not saved"
        assert (
            Path(test_params["base_results_dir"]) / "results.pkl"
        ).exists(), "Results not saved"


class TestBasePerformanceBenchmarking:
    """E2E tests for 02_Base_Performance_Benchmarking notebook.

    This test:
    1. Starts vLLM server externally
    2. Runs GuideLLM benchmark externally
    3. Runs notebook to verify results loading (with skipped CLI cells)

    Requires: base_model from TestBaseAccuracyBenchmarking
    """

    NOTEBOOK_DIR = "02_Base_Performance_Benchmarking"
    NOTEBOOK_NAME = "Base_Performance_Benchmarking.ipynb"

    @pytest.mark.e2e
    @pytest.mark.gpu
    @pytest.mark.timeout(3600)  # 1 hour
    @pytest.mark.requires_vllm
    @pytest.mark.requires_guidellm
    @pytest.mark.dependency(name="base_performance", depends=["base_accuracy"])
    def test_base_performance_benchmark(
        self,
        model_serve_flow_path,
        base_model_dir,
        compressed_model_dir,
        results_dir,
        executed_notebooks_dir,
        notebook_patcher,
        vllm_server_factory,
        guidellm_runner,
    ):
        """Run base performance benchmarking with external vLLM/GuideLLM."""
        notebook_path = model_serve_flow_path / self.NOTEBOOK_DIR / self.NOTEBOOK_NAME

        # Get model paths and check if base model exists
        model_name, base_path, _ = _get_model_paths(base_model_dir, compressed_model_dir)
        _skip_if_missing(
            base_path,
            "Base model",
            "TestBaseAccuracyBenchmarking::test_base_accuracy_full"
        )

        # Start vLLM server
        server = vllm_server_factory(
            model_path=base_path,
            port=8000,
            gpu_memory_utilization=0.6,
        )
        server_url = server.start()

        try:
            # Run GuideLLM benchmark
            benchmark_output = results_dir / "base_performance_benchmarks.json"
            guidellm_runner(
                target_url=server_url,
                output_path=benchmark_output,
                max_seconds=60,  # Reduced for testing
            )

            # Patch and run notebook for results loading/verification
            test_params = {
                "base_model_path": str(base_path),
                "vllm_target": server_url,
            }

            patcher = notebook_patcher(notebook_path)
            patcher.inject_parameters_cell(test_params)
            patcher.replace_hardcoded_paths(
                {
                    "base_model_path": "base_model_path",
                }
            )
            # Skip vLLM serve and GuideLLM benchmark cells (we ran them externally)
            patcher.skip_cells_matching(
                [
                    r"!pip install",
                    r"vllm serve",
                    r"guidellm benchmark",
                ]
            )

            patched_path = executed_notebooks_dir / f"patched_{self.NOTEBOOK_NAME}"
            patcher.save_to(patched_path)

            output_path = executed_notebooks_dir / f"executed_{self.NOTEBOOK_NAME}"
            pm.execute_notebook(
                str(patched_path),
                str(output_path),
                cwd=str(notebook_path.parent),
                kernel_name="python3",
            )

            # Verify benchmark results exist
            assert benchmark_output.exists(), "Benchmark results not created"
        finally:
            # Always stop server
            server.stop()


class TestModelCompression:
    """E2E tests for 03_Model_Compression notebook.

    Requires: base_model from TestBaseAccuracyBenchmarking
    """

    NOTEBOOK_DIR = "03_Model_Compression"
    NOTEBOOK_NAME = "Model_Compression.ipynb"

    @pytest.mark.e2e
    @pytest.mark.gpu
    @pytest.mark.timeout(7200)  # 2 hours
    @pytest.mark.dependency(name="compression", depends=["base_accuracy"])
    def test_model_compression(
        self,
        model_serve_flow_path,
        base_model_dir,
        compressed_model_dir,
        executed_notebooks_dir,
        notebook_patcher,
    ):
        """Run model compression with patched notebook."""
        notebook_path = model_serve_flow_path / self.NOTEBOOK_DIR / self.NOTEBOOK_NAME

        # Get model paths and check if base model exists
        model_name, base_path, compressed_path = _get_model_paths(
            base_model_dir, compressed_model_dir
        )
        _skip_if_missing(
            base_path,
            "Base model",
            "TestBaseAccuracyBenchmarking::test_base_accuracy_full"
        )

        test_params = {
            "base_model_path": str(base_path),
            "compressed_model_path": str(compressed_path),
            "num_calibration_samples": 64,  # Reduced for testing
            "max_sequence_length": 512,  # Reduced for testing
        }

        patcher = notebook_patcher(notebook_path)
        patcher.inject_parameters_cell(test_params)
        patcher.replace_hardcoded_paths(
            {
                "base_model_path": "base_model_path",
                "compressed_model_path": "compressed_model_path",
            }
        )
        patcher.skip_cells_matching([r"!pip install"])

        patched_path = executed_notebooks_dir / f"patched_{self.NOTEBOOK_NAME}"
        patcher.save_to(patched_path)

        output_path = executed_notebooks_dir / f"executed_{self.NOTEBOOK_NAME}"
        pm.execute_notebook(
            str(patched_path),
            str(output_path),
            cwd=str(notebook_path.parent),
            kernel_name="python3",
        )

        # Verify compressed model
        assert compressed_path.exists(), "Compressed model not created"
        assert (compressed_path / "config.json").exists(), "Model config not found"


class TestCompressedAccuracyBenchmarking:
    """E2E tests for 04_Compressed_Accuracy_Benchmarking notebook.

    Requires: compressed_model from TestModelCompression
    """

    NOTEBOOK_DIR = "04_Compressed_Accuracy_Benchmarking"
    NOTEBOOK_NAME = "Compressed_Accuracy_Benchmarking.ipynb"

    @pytest.mark.e2e
    @pytest.mark.gpu
    @pytest.mark.timeout(14400)  # 4 hours
    @pytest.mark.dependency(name="compressed_accuracy", depends=["compression"])
    def test_compressed_accuracy(
        self,
        model_serve_flow_path,
        base_model_dir,
        compressed_model_dir,
        results_dir,
        executed_notebooks_dir,
        notebook_patcher,
    ):
        """Run compressed accuracy benchmarking."""
        notebook_path = model_serve_flow_path / self.NOTEBOOK_DIR / self.NOTEBOOK_NAME

        # Get model paths and check if compressed model exists
        _, _, compressed_path = _get_model_paths(base_model_dir, compressed_model_dir)
        _skip_if_missing(
            compressed_path,
            "Compressed model",
            "TestModelCompression::test_model_compression"
        )

        test_params = {
            "compressed_model_path": str(compressed_path),
            "compressed_results_dir": str(results_dir / "compressed_accuracy"),
            "tasks": ["arc_easy"],
        }

        patcher = notebook_patcher(notebook_path)
        patcher.inject_parameters_cell(test_params)
        patcher.replace_hardcoded_paths(
            {
                "compressed_model_path": "compressed_model_path",
                "compressed_results_dir": "compressed_results_dir",
                "tasks": "tasks",
            }
        )
        patcher.skip_cells_matching([r"!pip install"])

        patched_path = executed_notebooks_dir / f"patched_{self.NOTEBOOK_NAME}"
        patcher.save_to(patched_path)

        output_path = executed_notebooks_dir / f"executed_{self.NOTEBOOK_NAME}"
        pm.execute_notebook(
            str(patched_path),
            str(output_path),
            cwd=str(notebook_path.parent),
            kernel_name="python3",
        )

        assert (Path(test_params["compressed_results_dir"]) / "results.pkl").exists()


class TestCompressedPerformanceBenchmarking:
    """E2E tests for 05_Compressed_Performance_Benchmarking notebook.

    Requires: compressed_model from TestModelCompression
    """

    NOTEBOOK_DIR = "05_Compressed_Performance_Benchmarking"
    NOTEBOOK_NAME = "Compressed_Performance_Benchmarking.ipynb"

    @pytest.mark.e2e
    @pytest.mark.gpu
    @pytest.mark.timeout(3600)  # 1 hour
    @pytest.mark.requires_vllm
    @pytest.mark.requires_guidellm
    @pytest.mark.dependency(name="compressed_performance", depends=["compression"])
    def test_compressed_performance_benchmark(
        self,
        model_serve_flow_path,
        base_model_dir,
        compressed_model_dir,
        results_dir,
        executed_notebooks_dir,
        notebook_patcher,
        vllm_server_factory,
        guidellm_runner,
    ):
        """Run compressed performance benchmarking."""
        notebook_path = model_serve_flow_path / self.NOTEBOOK_DIR / self.NOTEBOOK_NAME

        # Get model paths and check if compressed model exists
        _, _, compressed_path = _get_model_paths(base_model_dir, compressed_model_dir)
        _skip_if_missing(
            compressed_path,
            "Compressed model",
            "TestModelCompression::test_model_compression"
        )

        # Start vLLM server on different port
        server = vllm_server_factory(
            model_path=compressed_path,
            port=8001,
            gpu_memory_utilization=0.6,
        )
        server_url = server.start()

        try:
            # Run GuideLLM
            benchmark_output = results_dir / "compressed_performance_benchmarks.json"
            guidellm_runner(
                target_url=server_url,
                output_path=benchmark_output,
                max_seconds=60,
            )

            # Run notebook for verification
            test_params = {
                "compressed_model_path": str(compressed_path),
                "vllm_target": server_url,
            }

            patcher = notebook_patcher(notebook_path)
            patcher.inject_parameters_cell(test_params)
            patcher.replace_hardcoded_paths(
                {
                    "compressed_model_path": "compressed_model_path",
                }
            )
            patcher.skip_cells_matching(
                [
                    r"!pip install",
                    r"vllm serve",
                    r"guidellm benchmark",
                ]
            )

            patched_path = executed_notebooks_dir / f"patched_{self.NOTEBOOK_NAME}"
            patcher.save_to(patched_path)

            output_path = executed_notebooks_dir / f"executed_{self.NOTEBOOK_NAME}"
            pm.execute_notebook(
                str(patched_path),
                str(output_path),
                cwd=str(notebook_path.parent),
                kernel_name="python3",
            )

            assert benchmark_output.exists()
        finally:
            # Always stop server
            server.stop()
