# Synthetic Data Tooling

This module provides tools for generating synthetic training data for the planner model.

## Overview

The synthetic data generation pipeline converts solved reasoning traces into planning datasets. Given a problem and its solution, the system generates a "search plan" that a blind agent could follow to solve similar problems.

## Files

### `agnostic_data_gen.py`

Main script for generating synthetic planning data from reasoning traces.

**Data Source:**
- Uses `allenai/big-reasoning-traces` dataset (DeepSeek subset)
- Extracts problem-solution pairs
- Generates planning decompositions

**Usage:**
```bash
# Start vLLM server (using a capable model)
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --port 8000

# Run generation
python agnostic_data_gen.py
```

**Configuration (in-file):**
```python
OUTPUT_PATH = "synthetic_planning_dataset_local.jsonl"
MAX_EXAMPLES = 5  # Adjust for production runs
TEMPERATURE = 0.0  # Deterministic generation
```

**Output Format:**
```json
{
  "problem": "How does the nervous system regulate heart rate?",
  "plan": {
    "decompose": true,
    "subproblems": [
      "Identify the two main divisions of the autonomic nervous system...",
      "Determine the neurotransmitters released by the sympathetic system...",
      "Determine the neurotransmitters released by the parasympathetic system...",
      "Synthesize how these two systems interact..."
    ],
    "max_depth_hint": 1
  }
}
```

## Planner Prompt

The generation uses a carefully crafted prompt that instructs the model to:

1. **Convert to Interrogative/Imperative**: Subproblems must be questions or tasks, not facts
2. **Maximize Independence**: Subproblems should be solvable in parallel
3. **Determine Atomicity**: Simple fact retrieval → atomic; multi-step reasoning → decompose

### Example Transformations

**Good (Interrogative):**
```
BAD: "The medulla controls breathing."
GOOD: "Identify the specific brain structure responsible for breathing control."
```

**Math Example:**
```
Problem: A store sells 3 packs of gum for $2. How much for 15 packs?
Plan:
{
  "decompose": true,
  "subproblems": [
    "Determine the number of '3-pack groups' needed to make 15 packs.",
    "Calculate the total cost by multiplying the number of groups by price."
  ],
  "max_depth_hint": 1
}
```

## Safety Filter

The `is_safe_plan()` function filters out plans that require physical actions:

```python
physical_triggers = [
    "mix", "sample", "patient", "microscope", "dilute", "load", "weigh",
    "palpate", "surgery", "go to", "measure the"
]
```

Plans containing these terms are rejected since AI cannot perform physical actions.

## Key Functions

### `parse_json_response(text: str) -> dict`
Robustly extracts and parses JSON from LLM output:
- Strips markdown code blocks
- Handles trailing commas (common LLM error)
- Finds first valid JSON object

### `generate_plan(problem: str, reasoning: str) -> Optional[dict]`
Generates a planning decomposition:
- Formats the planner prompt
- Calls LLM via OpenAI-compatible API
- Returns parsed plan or None on error

### `is_safe_plan(plan: dict) -> bool`
Validates that a plan is safe for AI execution:
- Checks for physical action triggers
- Returns True for atomic plans (no subproblems)

## Output Statistics

```
Generation Complete.
Written: 1000
Skipped (Unsafe/Physical): 23
Skipped (Generation Errors): 12
Output saved to: synthetic_planning_dataset_local.jsonl
```

## Generated Dataset

The output JSONL file (`synthetic_planning_dataset_local.jsonl`) can be used for:
- Supervised Fine-Tuning (SFT) of planner models
- Creating training examples for GRPO
- Evaluating decomposition quality

## Dependencies

- `openai`: OpenAI API client
- `datasets`: HuggingFace datasets library
- `tqdm`: Progress bars
- `json`, `re`: JSON processing

## Notes

- Set `MAX_EXAMPLES` to control generation volume
- Use `TEMPERATURE = 0.0` for deterministic outputs
- Monitor skipped examples to tune safety filters
- Consider running multiple passes with different base datasets
