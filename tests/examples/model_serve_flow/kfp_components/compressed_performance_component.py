"""KFP component for compressed model performance benchmarking."""
import subprocess
import time
from pathlib import Path

from kfp import dsl
from kfp.dsl import Dataset, Output, Input

BASE_IMAGE = "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/pytorch:2024.2-py312-cuda"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "papermill",
        "nbformat",
        "ipykernel",
        "vllm",
        "guidellm",
        "openai",
        "requests",
    ],
)
def compressed_performance_benchmarking_component(
    compressed_model_input: Input[Dataset],
    performance_results_output: Output[Dataset],
    vllm_host: str = "127.0.0.1",
    vllm_port: int = 8001,
    benchmark_max_seconds: int = 120,
    gpu_memory_utilization: float = 0.6,
):
    """Execute compressed model performance benchmarking with vLLM and GuideLLM."""
    import sys
    import requests
    import papermill as pm

    # Add kfp_components to path for NotebookPatcher import
    sys.path.insert(0, "/opt/app-root/src/tests/examples/model_serve_flow")
    from notebook_patcher import NotebookPatcher

    compressed_model_path = Path(compressed_model_input.path)
    results_path = Path(performance_results_output.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Find actual model directory
    model_dirs = list(compressed_model_path.glob("*"))
    actual_model_path = model_dirs[0] if model_dirs else compressed_model_path

    # Start vLLM server
    vllm_cmd = [
        "vllm",
        "serve",
        str(actual_model_path),
        "--host",
        vllm_host,
        "--port",
        str(vllm_port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "2048",
    ]

    print(f"Starting vLLM server: {' '.join(vllm_cmd)}")
    vllm_proc = subprocess.Popen(
        vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    try:
        server_url = f"http://{vllm_host}:{vllm_port}"
        _wait_for_vllm(server_url, timeout=300)
        print(f"vLLM server ready at {server_url}")

        # Run GuideLLM benchmark
        output_file = results_path / "compressed_performance_benchmarks.json"
        guidellm_cmd = [
            "guidellm",
            "benchmark",
            "--target",
            server_url,
            "--profile",
            "sweep",
            "--max-seconds",
            str(benchmark_max_seconds),
            "--data",
            "prompt_tokens=1024,output_tokens=512",
            "--output-path",
            str(output_file),
        ]

        print(f"Running GuideLLM: {' '.join(guidellm_cmd)}")
        result = subprocess.run(guidellm_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"GuideLLM stderr: {result.stderr}")
            raise RuntimeError(f"GuideLLM failed with code {result.returncode}")

        # Patch and run notebook for verification
        notebook_dir = Path(
            "/opt/app-root/src/examples/model-serve-flow/05_Compressed_Performance_Benchmarking"
        )
        notebook_path = notebook_dir / "Compressed_Performance_Benchmarking.ipynb"

        test_params = {
            "compressed_model_path": str(actual_model_path),
            "vllm_target": server_url,
        }

        patcher = NotebookPatcher(notebook_path)
        patcher.inject_parameters_cell(test_params)
        patcher.replace_hardcoded_paths(
            {"compressed_model_path": "compressed_model_path"}
        )
        patcher.skip_cells_matching(
            [r"!pip install", r"vllm serve", r"guidellm benchmark"]
        )

        patched_path = Path("/tmp/patched_compressed_performance.ipynb")
        patcher.save_to(patched_path)

        pm.execute_notebook(
            str(patched_path),
            str(Path("/tmp/executed_compressed_performance.ipynb")),
            cwd=str(notebook_dir),
            kernel_name="python3",
        )

        print(f"Performance results saved to: {output_file}")

    finally:
        vllm_proc.terminate()
        vllm_proc.wait(timeout=30)


def _wait_for_vllm(url: str, timeout: int = 300):
    """Wait for vLLM server to be ready."""
    import requests

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")
