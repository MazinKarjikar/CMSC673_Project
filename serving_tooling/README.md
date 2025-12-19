# Serving Tooling

This module provides tools for deploying and serving the Recursive Reasoner models.

## Overview

The serving infrastructure supports:
- Interactive REPL interfaces for testing
- Model deployment to HuggingFace Hub
- Integration with vLLM for high-performance inference

## Files

### `push_to_hub.py`

Utility script for pushing models/adapters to HuggingFace Hub.

**Usage:**
```bash
# Push adapter configuration
python push_to_hub.py
```

### `repls/` Directory

Contains interactive REPL (Read-Eval-Print Loop) interfaces for testing the recursive reasoner.

#### `repl.py` - Main REPL with LoRA Planner

Full-featured REPL supporting trained planner with LoRA adapter.

**Usage:**
```bash
# Start vLLM with LoRA
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules planner=./adapter \
    --port 8000

# Run REPL
python repl.py --planner-model planner
```

**Command-Line Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--api-base` | `http://localhost:8000/v1` | vLLM API URL |
| `--planner-model` | `planner` | Planner model/adapter name |
| `--worker-model` | `meta-llama/Llama-3.1-8B-Instruct` | Worker model name |
| `--quiet` | `False` | Disable verbose output |

**REPL Commands:**
| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/models` | List available models |
| `/verbose` | Toggle verbose output |
| `/planner` | Show planner model |
| `/worker` | Show worker model |
| `/quit` | Exit |

#### `repl_base.py` - Baseline REPL

Uses the same base model for both planner and worker (no LoRA). Useful as a baseline for comparison.

**Usage:**
```bash
# Start vLLM (no LoRA needed)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Run baseline REPL
python repl_base.py
```

#### `repl_trained.py` - REPL with HuggingFace Model

Uses the trained planner model from HuggingFace Hub. Requires two vLLM servers.

**Usage:**
```bash
# Terminal 1: Start planner server (trained model)
python -m vllm.entrypoints.openai.api_server \
    --model vkaarti/recursive-reasoner-planner-llama3.1-8b \
    --port 8000

# Terminal 2: Start worker server (base model)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8001

# Terminal 3: Run REPL
python repl_trained.py
```

**Command-Line Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--planner-api` | `http://localhost:8000/v1` | Planner API URL |
| `--worker-api` | `http://localhost:8001/v1` | Worker API URL |
| `--planner-model` | `vkaarti/recursive-reasoner-planner-llama3.1-8b` | Planner model |
| `--worker-model` | `meta-llama/Llama-3.1-8B-Instruct` | Worker model |
| `--quiet` | `False` | Disable verbose output |

## Key Classes

### `VLLMClient`
Simple HTTP client for vLLM's OpenAI-compatible API:
- `generate()`: Generate completions using chat API
- `list_models()`: List available models on server

### `RecursiveReasoner`
Main reasoning class implementing the planner-worker architecture:
- `plan()`: Use planner to analyze problem
- `solve_subproblem()`: Solve individual subproblem
- `solve_atomic()`: Solve problem directly
- `combine()`: Combine subproblem solutions
- `solve()`: Main entry point

### `Colors`
ANSI color codes for terminal output formatting.

## UI Features

The REPL provides styled terminal output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  🧠  RECURSIVE REASONER                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

━━━ 📝 PROBLEM ━━━

Calculate 15% tip on a $85.50 bill split between 3 people.

━━━ 🧠 PHASE 1: PLANNING ━━━
✓ Decision: DECOMPOSE
Strategy: Multi-step calculation requiring tip and division.

┌────────────────────────────────────────────────────────────────────────┐
│                            SUBPROBLEMS                                  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Calculate 15% tip on $85.50                                         │
│ 2. Add tip to total                                                    │
│ 3. Divide by 3 people                                                  │
└────────────────────────────────────────────────────────────────────────┘

━━━ ⚙️  PHASE 2: SOLVING SUBPROBLEMS ━━━
...

━━━ ✅ FINAL ANSWER ━━━
┌────────────────────────────────────────────────────────────────────────┐
│                               RESULT                                    │
├────────────────────────────────────────────────────────────────────────┤
│ Each person pays $32.78                                                │
└────────────────────────────────────────────────────────────────────────┘
```

## Dependencies

- `requests`: HTTP client for API calls
- `argparse`: Command-line argument parsing
- `json`, `re`: JSON processing and regex

## Model Configurations

### Baseline Setup
- Single vLLM server
- Same model for planner and worker
- No LoRA adapters

### LoRA Setup
- Single vLLM server with LoRA enabled
- Planner uses LoRA adapter
- Worker uses base model

### Two-Server Setup
- Separate vLLM servers for planner and worker
- Planner: HuggingFace-hosted trained model
- Worker: Base model

## Notes

- Ensure vLLM server(s) are running before starting REPL
- Use `/verbose` to toggle detailed output
- The trained planner (`vkaarti/recursive-reasoner-planner-llama3.1-8b`) is publicly available
- API timeouts are set to 120 seconds for complex problems
