# Recursive Reasoner Tooling

This module contains the core infrastructure for recursive problem decomposition and solving.

## Overview

The recursive reasoner implements a hierarchical problem-solving approach where complex problems are decomposed into simpler subproblems, solved independently, and then combined. Two main architectures are provided:

1. **Multi-depth Recursion** (`infra.py`): Full recursive decomposition with configurable depth
2. **One-step Recursion** (`one_step_recurser.py`): Simpler planner-worker architecture with single decomposition level

## Files

### `infra.py`

Full recursive problem solver with multi-level decomposition.

**Features:**
- Configurable recursion depth (`max_depth`)
- Configurable subproblem width (`max_width`)
- Parallel subproblem solving using ThreadPoolExecutor
- Intelligent decomposition decisions

**Usage:**
```python
from infra import get_response, process_prompt

# Simple usage
response = get_response("Calculate 15% tip on $45.80 and split between 3 people")

# With verbose output
response, result = get_response("Your problem here", verbose=True)

# Advanced usage with custom parameters
result = process_prompt(client, model, "Your problem", max_depth=4, max_width=3)
```

**Key Functions:**

| Function | Description |
|----------|-------------|
| `get_response(prompt, max_depth=3, max_width=3, verbose=False)` | Main entry point for solving problems |
| `should_decompose(client, model, problem, depth, max_depth)` | Decides if a problem needs decomposition |
| `break_down_problem(client, model, problem, original_prompt, depth, max_width)` | Generates subproblems |
| `solve_atomic_problem(client, model, problem, original_prompt)` | Solves a single atomic problem |
| `combine_solutions(client, model, original_problem, subproblems, sub_solutions, original_prompt)` | Combines subproblem solutions |

**Decomposition Criteria:**
- DECOMPOSE: Problem explicitly asks for 2+ unrelated things
- ATOMIC: Single topic, sequential steps, or one domain

### `one_step_recurser.py`

Simplified one-step recursion with explicit planner-worker separation.

**Architecture:**
```
┌──────────────┐
│   PLANNER    │ → Decides: decompose or atomic
└──────────────┘   Generates: subproblems + plan
        │
        ▼
┌──────────────┐
│   WORKER     │ → Executes: subproblem solving
└──────────────┘   Combines: final answer
```

**Usage:**
```python
from one_step_recurser import get_response, one_step_solve

# Simple usage
response = get_response("Your math problem here", verbose=True)

# With result details
response, result = get_response("Problem", verbose=True)
print(f"Atomic: {result['is_atomic']}")
print(f"Plan: {result['plan']}")
print(f"Subproblems: {result['subproblems']}")
```

**Key Functions:**

| Function | Description |
|----------|-------------|
| `plan_problem(client, model, problem, max_width)` | PLANNER: Analyzes and generates plan |
| `solve_subproblem(client, model, subproblem, original_prompt, plan)` | WORKER: Solves single subproblem |
| `solve_atomic(client, model, problem, plan)` | WORKER: Solves atomically |
| `combine_solutions(client, model, original_problem, subproblems, sub_solutions, plan)` | WORKER: Combines solutions |
| `one_step_solve(client, planner_model, worker_model, problem, max_width)` | Main orchestration function |

## Interactive Mode

Both files can be run interactively:

```bash
# Start vLLM server first
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Run interactive mode
python infra.py
# or
python one_step_recurser.py
```

## Configuration

Both modules connect to vLLM server at:
- API Base: `http://localhost:8000/v1`
- API Key: `"EMPTY"` (vLLM default)

## JSON Response Parsing

The `parse_json_response()` function handles:
- Markdown code blocks (```json ... ```)
- Multi-line strings with actual newlines
- Malformed JSON with unescaped characters
- Multiple JSON candidates in output

## Decomposition Decision Logic

Problems are decomposed when ALL conditions are met:
1. Problem explicitly asks for 2+ unrelated things (AND, comma)
2. Each part requires completely different expertise
3. Parts can be solved independently with zero overlap

Problems are kept atomic when ANY applies:
- Single topic (even if complex)
- Sequential steps/process
- Single domain (ethics, math, science, etc.)
- Answer would be single coherent response

## Example Session

```
Enter your question: Calculate the area of a rectangle with length 5m and width 3m, AND explain the water cycle.

============================================================
Processing: Calculate the area of a rectangle with length 5m and width 3m...
============================================================

[PLANNER] Analyzing problem...
Planner decision: DECOMPOSE
Subproblems: ['Calculate rectangle area', 'Explain water cycle']
Plan: These are two independent topics requiring different approaches.

[WORKER] Solving 2 subproblems in parallel...
  Completed subproblem 1/2
  Completed subproblem 2/2

[WORKER] Combining solutions...

FINAL SOLUTION:
The area of the rectangle is 15 square meters (5m × 3m). The water cycle involves...
```

## Dependencies

- `openai`: OpenAI API client for vLLM compatibility
- `concurrent.futures`: Parallel execution of subproblems
- `json`, `re`: JSON parsing and text processing

## Notes

- Requires vLLM server running locally
- Use `verbose=True` for debugging and development
- Adjust `max_depth` and `max_width` based on problem complexity
- Higher `max_depth` allows more fine-grained decomposition but increases latency
