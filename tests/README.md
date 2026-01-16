# Testing Guide

This directory contains all tests for the Red Hat AI Examples repository.

For a complete overview of the testing infrastructure, see [TESTING.md](../TESTING.md) in the repository root.

## Quick Start

### Install Dependencies

```bash
# From the repository root
pip install -e ".[test]"
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Test Categories

| Test Suite | GPU Required | Description |
|------------|--------------|-------------|
| `tests/validation/` | ❌ No | Notebook structure, syntax, metadata validation |
| `tests/examples/knowledge_tuning/` | ❌ No | Knowledge tuning smoke tests |
| `tests/examples/model_serve_flow/` | ✅ Yes | Model compression & benchmarking E2E tests |

### Validation Tests (`tests/validation/`)

Repository-wide tests that validate all notebooks automatically. **No GPU required.**

```bash
# Run all validation tests
pytest tests/validation/ -v

# Run specific validation category
pytest tests/validation/test_notebook_structure.py -v
pytest tests/validation/test_notebook_content.py -v
pytest tests/validation/test_notebook_syntax.py -v
pytest tests/validation/test_notebook_metadata.py -v
pytest tests/validation/test_pyproject_toml.py -v
```

### Example-Specific Tests (`tests/examples/`)

Tests for specific examples in the repository.

#### Knowledge Tuning Tests

**No GPU required.**

```bash
# Run knowledge tuning smoke tests
pytest tests/examples/knowledge_tuning/ -v
```

#### Model Serve Flow E2E Tests

**⚠️ GPU Required** - These tests require a GPU with sufficient VRAM (~16GB+) to run the full model evaluation pipeline.

| Test | GPU | vLLM | GuideLLM | Description |
|------|-----|------|----------|-------------|
| `TestBaseAccuracyBenchmarking` | ✅ | ❌ | ❌ | Evaluates base model accuracy using lm-eval |
| `TestBasePerformanceBenchmarking` | ✅ | ✅ | ✅ | Benchmarks base model inference performance |
| `TestModelCompression` | ✅ | ❌ | ❌ | Compresses model using llmcompressor |
| `TestCompressedAccuracyBenchmarking` | ✅ | ❌ | ❌ | Evaluates compressed model accuracy |
| `TestCompressedPerformanceBenchmarking` | ✅ | ✅ | ✅ | Benchmarks compressed model performance |

---

### Detailed Test Descriptions

#### 1. TestBaseAccuracyBenchmarking

**Notebook:** `01_Base_Accuracy_Benchmarking/Base_Accuracy_Benchmarking.ipynb`

**What it does:**
- Downloads the base model from HuggingFace (`RedHatAI/Llama-3.1-8B-Instruct` by default)
- Saves the model locally for subsequent tests
- Runs accuracy evaluation using `lm-eval` on the `arc_easy` benchmark (reduced from full suite for faster testing)
- Saves evaluation results as `results.pkl`

**Pass criteria:**
- ✅ Notebook executes without errors
- ✅ Base model is saved to disk
- ✅ `results.pkl` file is created with evaluation metrics

**What passing means:**
- The notebook code is syntactically correct and executable
- Model loading and saving via `transformers` works correctly
- The `lm-eval` integration functions properly
- The evaluation pipeline produces valid output

**What is NOT tested:**
- ⚠️ Accuracy values are not validated (no threshold checks)
- ⚠️ Full benchmark suite (only `arc_easy`, not MMLU, IFEval, HellaSwag)
- ⚠️ Model quality or correctness of outputs

---

#### 2. TestBasePerformanceBenchmarking

**Notebook:** `02_Base_Performance_Benchmarking/Base_Performance_Benchmarking.ipynb`

**What it does:**
- Starts a vLLM server with the base model (programmatically, not via notebook)
- Runs GuideLLM benchmark for 60 seconds (reduced from production duration)
- Executes notebook with shell commands skipped (vLLM/GuideLLM run externally)
- Collects performance metrics (TTFT, ITL, throughput)

**Pass criteria:**
- ✅ vLLM server starts successfully and responds to health checks
- ✅ GuideLLM benchmark completes without errors
- ✅ Benchmark results JSON file is created
- ✅ Notebook executes without errors

**What passing means:**
- The base model can be served via vLLM
- The model responds to inference requests
- GuideLLM can successfully benchmark the model
- The notebook's analysis code works correctly

**What is NOT tested:**
- ⚠️ Performance thresholds (no min throughput or max latency checks)
- ⚠️ Extended load testing (only 60 seconds vs production workloads)
- ⚠️ Concurrent user simulation at scale
- ⚠️ Memory leak detection over time

---

#### 3. TestModelCompression

**Notebook:** `03_Model_Compression/Model_Compression.ipynb`

**What it does:**
- Loads the base model from the previous test
- Applies INT8 W8A8 quantization using `llmcompressor`
- Uses SmoothQuant + GPTQ modifiers with calibration data
- Saves the compressed model to disk
- Uses reduced calibration samples (64 vs 512) and sequence length (512 vs 1024) for faster testing

**Pass criteria:**
- ✅ Notebook executes without errors
- ✅ Compressed model directory is created
- ✅ `config.json` exists in the compressed model directory

**What passing means:**
- The quantization pipeline works end-to-end
- `llmcompressor` successfully compresses the model
- The compressed model has valid structure and configuration
- Calibration data loading and processing works correctly

**What is NOT tested:**
- ⚠️ Model size reduction (no check that compressed < base)
- ⚠️ Quantization quality (no perplexity or accuracy checks)
- ⚠️ Full calibration (reduced samples may not represent production quality)
- ⚠️ Different quantization schemes (only INT8 W8A8 tested)

---

#### 4. TestCompressedAccuracyBenchmarking

**Notebook:** `04_Compressed_Accuracy_Benchmarking/Compressed_Accuracy_Benchmarking.ipynb`

**What it does:**
- Loads the compressed model from the previous test
- Runs the same `lm-eval` benchmark (`arc_easy`) as the base model test
- Saves evaluation results as `results.pkl`

**Pass criteria:**
- ✅ Notebook executes without errors
- ✅ `results.pkl` file is created with evaluation metrics

**What passing means:**
- The compressed model can be loaded and used for inference
- The quantized model produces valid outputs
- The evaluation pipeline works with compressed models

**What is NOT tested:**
- ⚠️ Accuracy degradation (no comparison to base model results)
- ⚠️ Acceptable accuracy threshold (no pass/fail based on accuracy)
- ⚠️ Full benchmark suite (only `arc_easy`)

---

#### 5. TestCompressedPerformanceBenchmarking

**Notebook:** `05_Compressed_Performance_Benchmarking/Compressed_Performance_Benchmarking.ipynb`

**What it does:**
- Starts a vLLM server with the compressed model (on port 8001)
- Runs GuideLLM benchmark for 60 seconds
- Executes notebook with shell commands skipped
- Collects performance metrics

**Pass criteria:**
- ✅ vLLM server starts successfully with the compressed model
- ✅ GuideLLM benchmark completes without errors
- ✅ Benchmark results JSON file is created
- ✅ Notebook executes without errors

**What passing means:**
- The compressed model can be served via vLLM
- Quantized inference works correctly
- Performance benchmarking tools work with compressed models

**What is NOT tested:**
- ⚠️ Performance improvement (no comparison to base model)
- ⚠️ Throughput targets or latency SLAs
- ⚠️ Memory usage reduction verification

---

### Known Limitations & Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **No accuracy thresholds** | Tests don't fail if model accuracy drops significantly | Manual review of results.pkl files |
| **No performance comparison** | Tests don't compare base vs compressed metrics | Results saved for manual analysis |
| **Reduced test scope** | Uses `arc_easy` only, not full benchmark suite | Full benchmarks run in production |
| **Reduced calibration** | Compression uses fewer samples than recommended | May not reflect production quality |
| **No regression detection** | No baseline to compare against previous runs | Implement result tracking over time |
| **Single model tested** | Only tests default model, not all supported models | Run with different `TEST_MODEL_NAME` |
| **Time-limited benchmarks** | Performance tests run only 60 seconds | Extended benchmarks in production |

---

**Additional Dependencies:**

```bash
# Install ML dependencies for accuracy/compression tests (GPU required)
pip install papermill nbformat ipykernel
pip install torch transformers lm-eval accelerate llmcompressor datasets

# Install serving dependencies for performance tests (optional)
pip install vllm guidellm openai requests
```

> **Note:** Performance tests will be automatically skipped if vLLM or GuideLLM are not installed.

**Run Tests:**

```bash
# Run all model-serve-flow E2E tests (recommended - runs in dependency order)
pytest tests/examples/model_serve_flow/ -v

# Run individual tests (must respect dependency order)
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py::TestBaseAccuracyBenchmarking -v
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py::TestBasePerformanceBenchmarking -v
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py::TestModelCompression -v
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py::TestCompressedAccuracyBenchmarking -v
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py::TestCompressedPerformanceBenchmarking -v

# Run tests by pattern (matches multiple tests)
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py -v -k "Accuracy"      # Both accuracy tests
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py -v -k "Performance"   # Both performance tests
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py -v -k "Compression"   # Compression test only
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py -v -k "Base"          # Both base model tests
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py -v -k "Compressed"    # All compressed model tests
```

**Important:** Tests have dependencies. If you run `TestModelCompression` without first running `TestBaseAccuracyBenchmarking`, it will fail because the base model doesn't exist. Always run the full suite or respect the dependency order.

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `TEST_MODEL_NAME` | HuggingFace model to test | `RedHatAI/Llama-3.1-8B-Instruct` |
| `MODEL_SERVE_TEST_DIR` | Directory for test artifacts | `/tmp/model_serve_e2e_tests` |
| `CLEANUP_TEST_ARTIFACTS` | Clean up artifacts after session (set to `1`) | `0` (keep) |

> **Note:** Artifacts are kept by default to support running tests across multiple sessions.

**Clean up all artifacts:**

```bash
# Run the cleanup test to remove all downloaded models and results
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py::TestCleanup -v

# Or manually
rm -rf /tmp/model_serve_e2e_tests
```

**Test Dependency Graph:**

```text
TestBaseAccuracyBenchmarking (GPU) [~4 hours]
    ├── TestBasePerformanceBenchmarking (GPU + vLLM + GuideLLM) [~1 hour]
    └── TestModelCompression (GPU) [~2 hours]
            ├── TestCompressedAccuracyBenchmarking (GPU) [~4 hours]
            └── TestCompressedPerformanceBenchmarking (GPU + vLLM + GuideLLM) [~1 hour]
```

## Test Coverage

```bash
# Run tests with coverage report
pytest --cov=tests --cov-report=term

# Generate HTML coverage report
pytest --cov=tests --cov-report=html

# Show which lines are missing coverage
pytest --cov=tests --cov-report=term-missing
```

## Continuous Integration

Tests run automatically in GitHub Actions:

- **notebook-tests.yml**: Runs validation and smoke tests on every PR
- **model-serve-e2e-tests.yml**: Runs E2E tests on GPU runners (manual trigger or weekly)

## Adding New Tests

When adding new examples, they automatically undergo validation testing. For complex examples, consider adding smoke tests in `tests/examples/your-example-name/`.

See [TESTING.md](../TESTING.md) for detailed guidelines on test structure and requirements.
