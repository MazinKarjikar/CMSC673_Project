# Benchmark Tooling

This module provides tools for evaluating and comparing different model configurations on mathematical reasoning benchmarks.

## Overview

The benchmarking system compares three configurations:
1. **Original Model**: Direct solving with the base model (no decomposition)
2. **Base Recursive**: Base model used as both planner and worker
3. **Trained Recursive**: GRPO-trained planner + base model worker

## Files

### `benchmark.py`

Multi-dataset benchmarking tool supporting GSM8K, MATH, AQuA, and AIME datasets.

**Usage:**
```bash
# Start vLLM server with LoRA adapter
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules planner=./grpo_planner_adapter \
    --max-lora-rank 64 \
    --port 8000

# Run benchmarks
python benchmark.py --datasets gsm8k --samples 50
python benchmark.py --datasets gsm8k math --samples 100
python benchmark.py --datasets all --samples 200
```

**Command-Line Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | `gsm8k` | Datasets to benchmark (gsm8k, math, aqua, aime, all) |
| `--samples` | `50` | Number of samples per dataset |
| `--base-url` | `http://localhost:8000/v1` | vLLM API base URL |
| `--planner-adapter` | `planner` | Name of planner LoRA adapter |
| `--max-tokens` | `1024` | Max tokens for generation |
| `--temperature` | `0.6` | Sampling temperature |
| `--output-dir` | `benchmark_results` | Directory for results |
| `--quiet` | `False` | Less verbose output |

**Output:**
- Console summary with accuracy, decomposition rates, and timing
- JSON file with detailed results saved to `benchmark_results/`

### `gsm8k_compare.py`

Focused comparison tool for GSM8K dataset. Useful for quick A/B testing.

**Usage:**
```bash
python gsm8k_compare.py --samples 20
python gsm8k_compare.py --samples 50 --base-url http://localhost:8000/v1
```

**Command-Line Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--base-url` | `http://localhost:8000/v1` | vLLM API base URL |
| `--planner-adapter` | `planner` | Planner LoRA adapter name |
| `--samples` | `20` | Number of GSM8K samples |
| `--max-tokens` | `512` | Max tokens for generation |
| `--temperature` | `0.7` | Sampling temperature |
| `--quiet` | `False` | Less verbose output |

## Key Classes

### `BenchmarkConfig`
Configuration dataclass for benchmark runs containing API settings and generation parameters.

### `BenchmarkResult`
Stores results for a single benchmark run including:
- Accuracy metrics
- Decomposition statistics
- Timing information
- Detailed per-question results

### Solver Classes
- `OriginalSolver`: Direct solving without decomposition
- `BaseRecursiveSolver`: Uses base model for both planning and solving
- `TrainedRecursiveSolver`: Uses trained LoRA planner with base worker

## Key Functions

### `parse_json_response(output: str) -> Optional[Dict]`
Robustly parses JSON from model output, handling:
- Markdown code blocks
- Malformed JSON with newlines
- Multiple JSON objects in output

### `extract_number(text: str) -> Optional[float]`
Extracts numerical answers from text, handling:
- Currency symbols ($)
- Comma-separated numbers (1,000)
- Negative numbers

### `check_correct(predicted, ground_truth, tolerance=0.01) -> bool`
Checks if predicted answer matches ground truth:
- Exact match for integers
- Tolerance-based comparison for floats

### Dataset Loaders
- `load_gsm8k()`: Load OpenAI GSM8K dataset
- `load_math_dataset()`: Load Hendrycks MATH dataset
- `load_aqua_dataset()`: Load DeepMind AQuA-RAT dataset
- `load_aime_dataset()`: Load AIME problems

## Example Output

```
================================================================================
THREE-WAY BENCHMARK: Original vs Base Recursive vs Trained Recursive
================================================================================

DATASET: GSM8K
────────────────────────────────────────────────────────────────────────────────

▶ ORIGINAL
  [1/50] ✓ (A) [2.3s]
  [2/50] ✓ (A) [1.8s]
  ...

▶ BASE RECURSIVE
  [1/50] ✓ (D) [4.1s]
  ...

▶ TRAINED RECURSIVE
  [1/50] ✓ (D) [3.9s]
  ...

================================================================================
RESULTS SUMMARY
================================================================================

GSM8K:
Config                    Accuracy     Correct    Decomp%    Avg Time
----------------------------------------------------------------------
original                   72.0%      36/50       0.0%       2.15s
base_recursive             74.0%      37/50      68.0%       4.32s
trained_recursive          78.0%      39/50      72.0%       4.18s

ANALYSIS
----------------------------------------------------------------------
✓ Trained recursive is 6.0% better than original
```

## Dependencies

- `openai`: OpenAI API client
- `datasets`: HuggingFace datasets library
- `dataclasses`: Python dataclasses (standard library)

## Notes

- Ensure vLLM server is running before starting benchmarks
- Results are automatically saved with timestamps
- Use `--quiet` flag for batch processing
