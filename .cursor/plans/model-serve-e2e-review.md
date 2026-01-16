# Model Serve Flow E2E Plan Deep Review

## Scope
- Notebooks reviewed:
  - `examples/model-serve-flow/01_Base_Accuracy_Benchmarking/Base_Accuracy_Benchmarking.ipynb`
  - `examples/model-serve-flow/02_Base_Performance_Benchmarking/Base_Performance_Benchmarking.ipynb`
  - `examples/model-serve-flow/03_Model_Compression/Model_Compression.ipynb`
  - `examples/model-serve-flow/04_Compressed_Accuracy_Benchmarking/Compressed_Accuracy_Benchmarking.ipynb`
  - `examples/model-serve-flow/05_Compressed_Performance_Benchmarking/Compressed_Performance_Benchmarking.ipynb`
- Plan reviewed: `.cursor/plans/model_serve_e2e_tests_27b0ac97.plan.md`

## Summary of Alignment
The plan assumes fully automated papermill-driven execution with parameter overrides and deterministic output paths. The notebooks, however, are written for manual, interactive execution with hardcoded paths and external CLI steps for vLLM/GuideLLM. As written, the planned tests will not drive the notebooks into producing the expected artifacts without notebook changes or test adaptations.

## Findings

### High
1) **Papermill parameters will be ignored or overwritten by hardcoded notebook values.**
   - Base accuracy notebook sets `model_name`, `base_model_path`, `base_results_dir`, and `tasks` in code cells (hardcoded values).
   - Model compression notebook sets `base_model_path` and `compressed_model_path` directly.
   - Compressed accuracy notebook sets `compressed_model_path` and `compressed_results_dir` directly.
   - These cells appear after any papermill parameter injection, so test-supplied parameters (e.g., `base_model_path`, `compressed_model_path`, `base_results_dir`, `compressed_results_dir`, `tasks`) will not control output locations or model selection.
   - Impact: tests asserting outputs in temp `output_dir` will not find them because notebooks write to `../base_model`, `../compressed_model`, and `../results`.

2) **Performance notebooks depend on manual vLLM and GuideLLM CLI steps.**
   - Both performance notebooks instruct users to manually run `vllm serve` and `guidellm benchmark` in a terminal. The notebook cells only load results from `../results/*_performance_benchmarks.json`.
   - There is no notebook cell that runs the benchmarks programmatically, so papermill execution will not produce the JSON output the tests expect.
   - Impact: automated tests will fail because `output_path` is never created by the notebook itself.

### Medium
1) **Compression notebook pins incompatible Torch version.**
   - `Model_Compression.ipynb` installs `torch==2.9.0` while the notebook itself acknowledges `llmcompressor` requires `torch<=2.8.0`.
   - In automated CI, this is likely to cause runtime failures or unpredictable behavior.

2) **Planned parameter types differ from notebook usage.**
   - Plan expects `benchmark_max_seconds`, `num_calibration_samples`, and `max_sequence_length` to be injected and honored.
   - Notebook logic sets `num_calibration_samples` and `max_sequence_length` based on device and uses fixed values in the quantization call, so test overrides will not take effect.

3) **Output paths in notebooks do not align with planned test outputs.**
   - Base and compressed performance notebooks use `../results/*.json`.
   - Accuracy notebooks use `../results/base_accuracy` and `../results/compressed_accuracy`.
   - Plan expects outputs in a test-created temp directory; the current notebooks will bypass those paths.

### Low
1) **Manual GPU cleanup steps are embedded in notebook flow.**
   - Instructions to run `nvidia-smi` and `kill -9` are useful for human users but not automation.
   - These steps imply the notebooks were not designed for non-interactive test execution.

## Recommendations
1) **Introduce a papermill “parameters” cell in each notebook and use injected values consistently.**
   - Move or gate the hardcoded assignments so injected values are not overwritten.
   - Align with plan parameters: `model_name`, `base_model_path`, `compressed_model_path`, `base_results_dir`, `compressed_results_dir`, `benchmark_max_seconds`, `output_path`.

2) **Make performance notebooks self-contained for automated runs.**
   - Add a code cell that runs `vllm serve` and `guidellm benchmark` using subprocess within the notebook, or
   - Adjust tests to run those commands outside the notebook and only verify the results loading step.

3) **Align output paths with test expectations or update tests to match notebook defaults.**
   - If keeping notebook defaults, tests should assert in `../results`, `../base_model`, `../compressed_model`.
   - If keeping test defaults, notebooks must use injected paths.

4) **Reconcile dependency versions for CI stability.**
   - Avoid forcing `torch==2.9.0` in notebook installs unless the base image and `llmcompressor` support it.
   - Prefer using repo-managed dependencies or version pins compatible with `llmcompressor`.
