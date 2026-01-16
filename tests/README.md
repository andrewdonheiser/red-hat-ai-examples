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

### Validation Tests (`tests/validation/`)

Repository-wide tests that validate all notebooks automatically. No GPU required.

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

```bash
# Run knowledge tuning smoke tests (no GPU required)
pytest tests/examples/knowledge_tuning/ -v
```

#### Model Serve Flow E2E Tests

These tests require a GPU with sufficient VRAM (~16GB+) to run the full model evaluation pipeline.

**Additional Dependencies:**

```bash
# Install ML dependencies for accuracy tests
pip install papermill nbformat ipykernel
pip install torch transformers lm-eval accelerate llmcompressor datasets

# Install serving dependencies for performance tests
pip install vllm guidellm openai requests
```

**Run Tests:**

```bash
# Run all model-serve-flow E2E tests
pytest tests/examples/model_serve_flow/ -v

# Run only accuracy benchmarking tests
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py -v -k "Accuracy"

# Run only performance benchmarking tests
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py -v -k "Performance"

# Run model compression test
pytest tests/examples/model_serve_flow/test_e2e_notebooks.py -v -k "Compression"
```

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `TEST_MODEL_NAME` | HuggingFace model to test | `RedHatAI/Llama-3.1-8B-Instruct` |
| `KEEP_TEST_ARTIFACTS` | Keep model artifacts after tests (set to `1`) | `0` (cleanup) |

**Test Dependency Graph:**

```text
test_base_accuracy_full
    ├── test_base_performance_benchmark (needs base_model)
    └── test_model_compression (needs base_model)
            ├── test_compressed_accuracy (needs compressed_model)
            └── test_compressed_performance_benchmark (needs compressed_model)
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
