"""Pytest fixtures for model-serve-flow e2e tests.

Note: The `repo_root` fixture is inherited from tests/conftest.py
"""
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest
import requests

from .notebook_patcher import NotebookPatcher, NOTEBOOK_PATCHES


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "e2e: end-to-end test")
    config.addinivalue_line("markers", "gpu: requires GPU resources")
    config.addinivalue_line("markers", "requires_vllm: requires vLLM server")
    config.addinivalue_line("markers", "requires_guidellm: requires GuideLLM")
    config.addinivalue_line("markers", "dependency: test dependency marker")
    config.addinivalue_line("markers", "timeout: test timeout in seconds")


def _is_tool_available(tool_name: str) -> bool:
    """Check if a command-line tool is available."""
    try:
        # vLLM uses --help, not --version
        flag = "--help" if tool_name == "vllm" else "--version"
        result = subprocess.run(
            [tool_name, flag],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available tools."""
    # Check tool availability once
    vllm_available = _is_tool_available("vllm")
    guidellm_available = _is_tool_available("guidellm")

    skip_vllm = pytest.mark.skip(reason="vLLM not installed")
    skip_guidellm = pytest.mark.skip(reason="GuideLLM not installed")

    for item in items:
        if "requires_vllm" in item.keywords and not vllm_available:
            item.add_marker(skip_vllm)
        if "requires_guidellm" in item.keywords and not guidellm_available:
            item.add_marker(skip_guidellm)


# Default timeout for e2e tests (4 hours for accuracy, 1 hour for performance)
DEFAULT_ACCURACY_TIMEOUT = 4 * 60 * 60  # 4 hours in seconds
DEFAULT_PERFORMANCE_TIMEOUT = 60 * 60  # 1 hour in seconds


@pytest.fixture(scope="session")
def model_serve_flow_path(repo_root) -> Path:
    """Path to model-serve-flow example."""
    return repo_root / "examples" / "model-serve-flow"


@pytest.fixture(scope="session")
def test_output_dir() -> Path:
    """Session-scoped directory for all test outputs.

    Uses a persistent location so artifacts survive across pytest sessions.
    Override with MODEL_SERVE_TEST_DIR environment variable.
    """
    # Allow override via environment variable
    env_path = os.environ.get("MODEL_SERVE_TEST_DIR")
    if env_path:
        path = Path(env_path)
    else:
        # Use a fixed location that persists across sessions
        path = Path("/tmp/model_serve_e2e_tests")

    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def base_model_dir(test_output_dir) -> Path:
    """Directory for base model artifacts."""
    path = test_output_dir / "base_model"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def compressed_model_dir(test_output_dir) -> Path:
    """Directory for compressed model artifacts."""
    path = test_output_dir / "compressed_model"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def results_dir(test_output_dir) -> Path:
    """Directory for benchmark results."""
    path = test_output_dir / "results"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def executed_notebooks_dir(test_output_dir) -> Path:
    """Directory for executed notebook outputs."""
    path = test_output_dir / "executed_notebooks"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture
def notebook_patcher():
    """Factory fixture for creating NotebookPatcher instances."""

    def _create_patcher(notebook_path: Path) -> NotebookPatcher:
        return NotebookPatcher(notebook_path)

    return _create_patcher


def _cleanup_gpu_processes():
    """Kill any leftover GPU processes to free memory.

    This helps prevent OOM errors when starting vLLM.
    """
    try:
        # Get list of GPU processes
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            print("Warning: Could not query GPU processes")
            return

        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]

        if not pids:
            print("No GPU processes found")
            return

        print(f"Found {len(pids)} GPU process(es): {pids}")

        # Show what's using GPU memory
        subprocess.run(["nvidia-smi"], timeout=10)

        # Kill each process
        for pid in pids:
            try:
                print(f"Killing GPU process {pid}...")
                subprocess.run(["kill", "-9", pid], timeout=5)
            except Exception as e:
                print(f"Warning: Could not kill process {pid}: {e}")

        # Wait a moment for processes to terminate
        time.sleep(2)

        print("GPU cleanup complete")

    except FileNotFoundError:
        print("nvidia-smi not found - skipping GPU cleanup")
    except Exception as e:
        print(f"Warning: GPU cleanup failed: {e}")


class VLLMServer:
    """Context manager for vLLM server lifecycle."""

    def __init__(
        self,
        model_path: Path,
        host: str = "127.0.0.1",
        port: int = 8000,
        gpu_memory_utilization: float = 0.6,
        max_model_len: int = 2048,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.process = None
        self.url = f"http://{host}:{port}"

    def start(self, timeout: int = 300) -> str:
        """Start vLLM server and wait for it to be ready."""
        # Clean up any leftover GPU processes first
        _cleanup_gpu_processes()

        # Use exact same parameters as the notebook:
        # vllm serve "../base_model/..." --host 127.0.0.1 --port 8000 \
        #   --gpu-memory-utilization 0.6 --tensor-parallel-size 1 \
        #   --pipeline-parallel-size 1 --max-model-len 2048
        cmd = [
            "vllm",
            "serve",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            "1",
            "--pipeline-parallel-size",
            "1",
            "--max-model-len",
            str(self.max_model_len),
        ]

        print(f"Starting vLLM server: {' '.join(cmd)}")

        # Run vLLM like in the notebook - output to terminal, not captured
        # Capturing stdout can cause buffer issues that affect server behavior
        self.log_file = open("/tmp/vllm_server.log", "w")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            # Don't buffer output
            bufsize=0,
        )

        self._wait_for_ready(timeout)
        print(f"vLLM server ready at {self.url}")
        return self.url

    def _wait_for_ready(self, timeout: int):
        """Wait for server to be fully ready.

        Checks both health endpoint and models endpoint, then waits
        for stabilization before returning.
        """
        start = time.time()

        # First, wait for health endpoint
        print("Waiting for vLLM health endpoint...")
        while time.time() - start < timeout:
            # Check if process crashed
            if self.process.poll() is not None:
                # Read output from log file for debugging
                output = ""
                try:
                    with open("/tmp/vllm_server.log", "r") as f:
                        output = f.read()
                except Exception:
                    pass
                raise RuntimeError(
                    f"vLLM server process exited with code {self.process.returncode}\n"
                    f"Server output:\n{output}"
                )

            try:
                resp = requests.get(f"{self.url}/health", timeout=5)
                if resp.status_code == 200:
                    print("Health endpoint ready")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(5)
        else:
            raise TimeoutError(f"vLLM health endpoint not ready after {timeout}s")

        # Then, wait for models endpoint (required by GuideLLM)
        print("Waiting for vLLM models endpoint...")
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self.url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    print("Models endpoint ready")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"vLLM models endpoint not ready after {timeout}s")

        # Give server additional time to stabilize
        print("Waiting 10s for server to stabilize...")
        time.sleep(10)
        print("vLLM server fully ready")

    def stop(self):
        """Stop the vLLM server."""
        if self.process:
            print("Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        # Close log file
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
            self.log_file = None


@pytest.fixture
def vllm_server_factory():
    """Factory fixture for creating vLLM server instances."""
    servers = []

    def _create_server(**kwargs) -> VLLMServer:
        server = VLLMServer(**kwargs)
        servers.append(server)
        return server

    yield _create_server

    # Cleanup all servers
    for server in servers:
        server.stop()


def run_guidellm_benchmark(
    target_url: str,
    output_path: Path,
    max_seconds: int = 120,  # Same as notebook
    prompt_tokens: int = 1024,
    output_tokens: int = 512,
) -> Path:
    """Run GuideLLM benchmark and return results path.

    Uses the exact same parameters as the notebook:
    guidellm benchmark --target "http://127.0.0.1:8000" --profile sweep \
        --max-seconds 120 --data "prompt_tokens=1024,output_tokens=512"
    """
    # Verify server is still responding before starting benchmark
    print(f"Verifying vLLM server at {target_url}...")
    try:
        resp = requests.get(f"{target_url}/v1/models", timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM server not responding: {resp.status_code}")
        models = resp.json()
        print(f"vLLM models available: {models}")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to vLLM server: {e}")

    # Set GUIDELLM_NUM_WORKERS=1 to reduce concurrent connections
    env = os.environ.copy()
    env["GUIDELLM_NUM_WORKERS"] = "1"

    # Use exact same parameters as notebook (no --rate flag)
    # Generous timeout: 1 hour to handle slower hardware
    # Default sweep runs 10 benchmarks, each for max_seconds
    timeout_seconds = 3600  # 1 hour

    cmd = [
        "guidellm",
        "benchmark",
        "--target",
        target_url,
        "--profile", "sweep",
        "--max-seconds",
        str(max_seconds),
        "--data",
        f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
        "--output-path",
        str(output_path),
    ]

    print(f"Running GuideLLM: {' '.join(cmd)}")
    print(f"Environment: GUIDELLM_NUM_WORKERS=1")
    print(f"Timeout: {timeout_seconds}s (1 hour)")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, env=env)

    if result.returncode != 0:
        print(f"GuideLLM stdout: {result.stdout}")
        print(f"GuideLLM stderr: {result.stderr}")
        raise RuntimeError(f"GuideLLM failed with code {result.returncode}")

    print(f"GuideLLM completed successfully")
    return output_path


@pytest.fixture
def guidellm_runner():
    """Fixture providing GuideLLM benchmark runner."""
    return run_guidellm_benchmark


@pytest.fixture(scope="session", autouse=True)
def cleanup_artifacts(test_output_dir):
    """Clean up test artifacts after test session completes.

    By default, artifacts are KEPT to support running tests across sessions.
    Set CLEANUP_TEST_ARTIFACTS=1 to clean up after the session.
    """
    yield

    if os.environ.get("CLEANUP_TEST_ARTIFACTS", "0") != "1":
        print(f"Keeping test artifacts at: {test_output_dir}")
        print("Set CLEANUP_TEST_ARTIFACTS=1 to clean up after tests.")
        return

    # Clean up large model artifacts to prevent disk space issues
    print(f"Cleaning up test artifacts at: {test_output_dir}")
    try:
        shutil.rmtree(test_output_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to clean up artifacts: {e}")


@pytest.fixture
def artifact_dir_from_env(test_output_dir) -> Path:
    """Get artifact directory from environment or use default.

    This fixture allows performance tests to find artifacts downloaded
    from previous job's GitHub artifacts.
    """
    env_path = os.environ.get("MODEL_ARTIFACTS_DIR")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return test_output_dir
