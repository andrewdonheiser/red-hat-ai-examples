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
        result = subprocess.run(
            [tool_name, "--version"],
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
def test_output_dir(tmp_path_factory) -> Path:
    """Session-scoped directory for all test outputs."""
    return tmp_path_factory.mktemp("model_serve_e2e")


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
            "--max-model-len",
            str(self.max_model_len),
        ]

        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self._wait_for_ready(timeout)
        print(f"vLLM server ready at {self.url}")
        return self.url

    def _wait_for_ready(self, timeout: int):
        """Wait for server health endpoint."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self.url}/health", timeout=5)
                if resp.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM server not ready after {timeout}s")

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
    max_seconds: int = 120,
    prompt_tokens: int = 1024,
    output_tokens: int = 512,
) -> Path:
    """Run GuideLLM benchmark and return results path."""
    cmd = [
        "guidellm",
        "benchmark",
        "--target",
        target_url,
        "--profile",
        "sweep",
        "--max-seconds",
        str(max_seconds),
        "--data",
        f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
        "--output-path",
        str(output_path),
    ]

    print(f"Running GuideLLM: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"GuideLLM stdout: {result.stdout}")
        print(f"GuideLLM stderr: {result.stderr}")
        raise RuntimeError(f"GuideLLM failed with code {result.returncode}")

    return output_path


@pytest.fixture
def guidellm_runner():
    """Fixture providing GuideLLM benchmark runner."""
    return run_guidellm_benchmark


@pytest.fixture(scope="session", autouse=True)
def cleanup_artifacts(test_output_dir):
    """Clean up test artifacts after test session completes.

    Artifacts are preserved if KEEP_TEST_ARTIFACTS=1 is set (useful for debugging).
    """
    yield

    if os.environ.get("KEEP_TEST_ARTIFACTS", "0") == "1":
        print(f"Keeping test artifacts at: {test_output_dir}")
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
