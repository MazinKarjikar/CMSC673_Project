# Recursive Reasoner

A recursive reasoning system that decomposes complex math problems into subproblems, solves them individually, and combines the solutions.

## Three Versions

| Script | Planner | Worker | Purpose |
|--------|---------|--------|---------|
| `repl.py` | LoRA adapter | Base Llama 3.1 | Development with local adapter |
| `repl_base.py` | Base Llama 3.1 | Base Llama 3.1 | **Baseline** comparison |
| `repl_trained.py` | HuggingFace trained | Base Llama 3.1 | **Production** with trained planner |

## Quick Start

### Option 1: Use the Trained Planner from HuggingFace (Recommended)

```bash
# Terminal 1: Start vLLM with the trained planner (port 8000)
python -m vllm.entrypoints.openai.api_server \
    --model vkaarti/recursive-reasoner-planner-llama3.1-8b \
    --port 8000

# Terminal 2: Start vLLM with the base worker (port 8001)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8001

# Terminal 3: Run the trained REPL
python repl_trained.py
```

### Option 2: Baseline (Compare Performance)

```bash
# Start vLLM with base model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Run the baseline REPL
python repl_base.py
```

### Option 3: Development with Local LoRA Adapter

```bash
# Start vLLM with LoRA adapter
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules planner=./grpo_planner_adapter \
    --max-lora-rank 32 \
    --port 8000

# Run the development REPL
python repl.py
```

## Usage

Once the REPL is running, simply type a math problem:

```
>>> Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins. She sells the rest for $2 each. How much does she make daily?

━━━ 📝 PROBLEM ━━━
Janet's ducks lay 16 eggs per day...

━━━ 🧠 PHASE 1: PLANNING ━━━
✓ Decision: DECOMPOSE
Strategy: Calculate remaining eggs, then compute revenue

┌────────────────────────────────────────┐
│            SUBPROBLEMS                 │
├────────────────────────────────────────┤
│ 1. Calculate eggs consumed             │
│ 2. Calculate remaining eggs            │
│ 3. Calculate revenue                   │
└────────────────────────────────────────┘

━━━ ⚙️  PHASE 2: SOLVING SUBPROBLEMS ━━━
...

━━━ ✅ FINAL ANSWER ━━━
┌────────────────────────────────────────┐
│               RESULT                   │
├────────────────────────────────────────┤
│ Final Answer: 18                       │
└────────────────────────────────────────┘
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/models` | List available models |
| `/verbose` | Toggle verbose output |
| `/planner` | Show planner model |
| `/worker` | Show worker model |
| `/quit` | Exit |

## Requirements

```bash
pip install requests
```

## Model Information

**Trained Planner**: [vkaarti/recursive-reasoner-planner-llama3.1-8b](https://huggingface.co/vkaarti/recursive-reasoner-planner-llama3.1-8b)

The planner was trained using GRPO (Group Relative Policy Optimization) on the GSM8K dataset to learn effective problem decomposition strategies.

## LoRA Adapters

Pre-trained LoRA adapters are available for download:

| Adapter | Description | Download |
|---------|-------------|----------|
| `grpo_planner_adapter` | GRPO-trained planner | [Google Drive](https://drive.google.com/{TODO}) |

To use a downloaded adapter:
1. Extract the `grpo_planner_adapter` folder to this directory
2. Start vLLM with LoRA enabled (see Option 3 above)
3. Run `python repl.py`

