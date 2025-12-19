# CMSC673 Machine Learning Capstone Project: Recursive Reasoner

## Overview

This project implements a **Recursive Reasoning System** for mathematical problem-solving using Large Language Models (LLMs). The system uses a novel planner-worker architecture where:

- **Planner Model**: Analyzes problems and decides whether to decompose them into subproblems or solve directly
- **Worker Model**: Executes the actual problem-solving, either atomically or by solving subproblems and combining results

The planner is trained using **Group Relative Policy Optimization (GRPO)** with LoRA adapters, enabling efficient fine-tuning while keeping the base model frozen.

## Team

- Mazin Karjikar
- Omkar Pathak
- Vijaykaarti Sundarapandiyan
- Aditya Menachery

## Project Structure

```
CMSC673_Project/
├── benchmark_tooling/          # Benchmarking and evaluation tools
│   ├── benchmark.py           # Multi-dataset benchmarking (GSM8K, MATH, AQuA, AIME)
│   └── gsm8k_compare.py       # GSM8K-specific comparison tool
│
├── recursive_reasoner_tooling/ # Core recursive reasoning infrastructure
│   ├── infra.py               # Multi-depth recursive problem solver
│   └── one_step_recurser.py   # One-step planner-worker architecture
│
├── synthetic_data_tooling/     # Synthetic training data generation
│   └── agnostic_data_gen.py   # Generate planning data from reasoning traces
│
├── training_tooling/           # Model training scripts
│   ├── grpo_training.py       # GRPO training with LoRA for planner
│   ├── dummy_planner.py       # SFT training baseline
│   ├── inference_vllm.py      # vLLM inference with LoRA adapters
│   └── merge_and_push.py      # Merge LoRA weights and push to HuggingFace
│
├── serving_tooling/            # Model serving and deployment
│   ├── push_to_hub.py         # Push models to HuggingFace Hub
│   └── repls/                 # Interactive REPL interfaces
│       ├── repl.py           # Main REPL with trained planner
│       ├── repl_base.py      # Baseline REPL (untrained planner)
│       └── repl_trained.py   # REPL with HuggingFace-hosted planner
│
└── requirements.txt            # Project dependencies
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended: 24GB+ VRAM for training)
- [vLLM](https://github.com/vllm-project/vllm) for efficient inference

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd CMSC673_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Start vLLM Server

For baseline evaluation (untrained planner):
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

For trained planner with LoRA:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules planner=./grpo_planner_adapter \
    --max-lora-rank 64 \
    --port 8000
```

### 2. Interactive REPL

```bash
# With trained planner
python serving_tooling/repls/repl.py

# Baseline (untrained)
python serving_tooling/repls/repl_base.py
```

### 3. Run Benchmarks

```bash
# Benchmark on GSM8K (50 samples)
python benchmark_tooling/benchmark.py --datasets gsm8k --samples 50

# Benchmark on all datasets
python benchmark_tooling/benchmark.py --datasets all --samples 100
```

## Training

### GRPO Training (Recommended)

Train the planner model using Group Relative Policy Optimization:

```bash
python training_tooling/grpo_training.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --max-steps 500 \
    --lora-r 16 \
    --lora-alpha 32 \
    --output-dir ./grpo_planner_adapter
```

### Reward Function

The GRPO training uses a composite reward:
- **Correctness** (weight=1.0): Did the pipeline produce the correct answer?
- **Token Efficiency** (weight=0.2): Penalty for verbose solutions
- **Independence** (weight=0.3): Reward for generating independent subproblems

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Problem                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PLANNER (LoRA-trained)                   │
│  - Analyzes problem complexity                              │
│  - Decides: DECOMPOSE or ATOMIC                             │
│  - If decompose: generates subproblems                      │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│     ATOMIC PATH         │     │    DECOMPOSED PATH      │
│                         │     │                         │
│  Worker solves directly │     │  1. Worker solves each  │
│                         │     │     subproblem          │
│                         │     │  2. Worker combines     │
│                         │     │     solutions           │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final Answer                              │
└─────────────────────────────────────────────────────────────┘
```

## Supported Benchmarks

| Dataset | Description | Size |
|---------|-------------|------|
| GSM8K | Grade school math problems | 8.5K |
| MATH | Competition mathematics | 12.5K |
| AQuA | Algebraic word problems | 100K |
| AIME | American Invitational Math Exam | 900+ |

## Model Artifacts

Trained models are available on HuggingFace and Drive:
- Planner: `vkaarti/recursive-reasoner-planner-llama3.1-8b`
- LoRA adapter on [Drive](https://drive.google.com/file/d/1q5AcdpQvCgMLdJmJCkMW6VOddiNz0eVi/view?usp=sharing)

## Testing

Run unit tests:
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_benchmark.py -v
```

## Configuration

Key configuration options are available via command-line arguments. See each module's `--help` for details:

```bash
python benchmark_tooling/benchmark.py --help
python training_tooling/grpo_training.py --help
```

## License

This project was created for CMSC673 at the University of Maryland.

## Acknowledgments

- Built on top of [Llama 3.1](https://ai.meta.com/llama/)
- Training framework: [TRL](https://github.com/huggingface/trl)
- Inference: [vLLM](https://github.com/vllm-project/vllm)
- Adapter method: [LoRA](https://arxiv.org/abs/2106.09685)
