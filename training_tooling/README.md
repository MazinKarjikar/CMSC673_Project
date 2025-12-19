# Training Tooling

This module contains all model training scripts for the Recursive Reasoner project.

## Overview

The training pipeline uses:
- **GRPO (Group Relative Policy Optimization)**: For reinforcement learning-based planner training
- **LoRA (Low-Rank Adaptation)**: For parameter-efficient fine-tuning
- **vLLM**: For efficient inference during training and deployment

## Files

### `grpo_training.py`

Main training script using GRPO to train the planner model.

**Architecture:**
- **Planner (Trained)**: LoRA adapter on base model, learns to decompose problems
- **Worker (Frozen)**: Base model, executes subproblems and combines solutions

**Usage:**
```bash
python grpo_training.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --max-steps 500 \
    --lora-r 16 \
    --lora-alpha 32 \
    --output-dir ./grpo_planner_adapter
```

**Command-Line Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--base-model` | `meta-llama/Llama-3.1-8B-Instruct` | Base model name |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lr` | `1e-5` | Learning rate |
| `--batch-size` | `1` | Per-device batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--max-steps` | `500` | Maximum training steps |
| `--num-generations` | `4` | GRPO generations per step |
| `--temperature` | `0.7` | Sampling temperature |
| `--correctness-weight` | `1.0` | Weight for correctness reward |
| `--token-weight` | `0.2` | Weight for token efficiency |
| `--independence-weight` | `0.3` | Weight for subproblem independence |
| `--max-samples` | `1000` | Dataset size |
| `--output-dir` | `./grpo_planner_adapter` | Output directory |
| `--wandb-project` | `recursive-reasoner-grpo` | W&B project name |
| `--no-wandb` | `False` | Disable W&B logging |
| `--no-4bit` | `False` | Disable 4-bit quantization |

**Reward Function:**
```
reward = correctness_weight × correct 
       + token_efficiency_weight × (1 - tokens/max_tokens)
       + independence_weight × independence_score
```

Where:
- `correct`: 1.0 if answer matches ground truth, 0.0 otherwise
- `token_efficiency`: Inversely proportional to total tokens used
- `independence_score`: Semantic dissimilarity between subproblems (using sentence embeddings)

### `dummy_planner.py`

Supervised Fine-Tuning (SFT) training baseline.

**Usage:**
```bash
python dummy_planner.py
```

**Configuration (in-file):**
```python
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "emilbiju/Planning-Data-Math-Full-Soln"
OUTPUT_DIR = "./deepseek-r1-planner-lora"

MAX_SEQ_LEN = 17000
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 1e-4
NUM_EPOCHS = 1.0
```

### `inference_vllm.py`

Inference script for using trained models with vLLM.

**Usage:**
```bash
# Start vLLM server with LoRA
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules recursive_reasoner=./grpo_lora_adapters \
    --port 8000

# Run inference
python inference_vllm.py --lora-adapter recursive_reasoner
```

**Command-Line Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--base-url` | `http://localhost:8000/v1` | vLLM API URL |
| `--model` | `None` | Model name (auto-detected) |
| `--lora-adapter` | `recursive_reasoner` | LoRA adapter name |
| `--max-tokens` | `512` | Max tokens |
| `--temperature` | `0.7` | Sampling temperature |

### `merge_and_push.py`

Merge LoRA adapter with base model and push to HuggingFace Hub.

**Usage:**
```bash
# Merge and push to HuggingFace
python merge_and_push.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --adapter-path ./grpo_planner_adapter \
    --repo-name your-username/model-name

# Save locally only
python merge_and_push.py \
    --adapter-path ./adapter \
    --repo-name test/model \
    --local-save-path ./merged_model \
    --no-push
```

**Command-Line Arguments:**
| Argument | Description |
|----------|-------------|
| `--base-model` | Base model name |
| `--adapter-path` | Path to LoRA adapter |
| `--repo-name` | HuggingFace repo name (required) |
| `--model-card` | Optional model card path |
| `--local-save-path` | Local save path |
| `--no-push` | Don't push to Hub |
| `--private` | Make repo private |

## Key Classes

### `TrainingArgs` (grpo_training.py)
Dataclass containing all training configuration parameters.

### `FrozenWorker` (grpo_training.py)
Frozen worker model for reward computation:
- `solve_subproblem()`: Solves individual subproblems
- `combine_solutions()`: Combines subproblem solutions
- `solve_directly()`: Atomic problem solving

### `PipelineExecutor` (grpo_training.py)
Executes full pipeline for reward computation:
- `parse_planner_output()`: Parses planner JSON output
- `execute_pipeline()`: Runs complete pipeline
- `compute_independence()`: Calculates subproblem independence score

### `PipelineRewardFunction` (grpo_training.py)
Computes rewards for GRPO training:
- Executes pipeline for each planner completion
- Computes composite reward
- Logs metrics to W&B

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    GSM8K Dataset                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Planner (LoRA)                           │
│  Generates: should_decompose, subproblems, plan             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Worker (Frozen)                          │
│  Executes: solve subproblems → combine → final answer       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Reward Computation                        │
│  - Correctness: answer == ground_truth                      │
│  - Efficiency: tokens used                                  │
│  - Independence: subproblem similarity                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Update                              │
│  Update planner LoRA weights based on reward                │
└─────────────────────────────────────────────────────────────┘
```

## Dependencies

- `torch`: PyTorch
- `transformers`: HuggingFace Transformers
- `peft`: Parameter-Efficient Fine-Tuning
- `trl`: Transformer Reinforcement Learning
- `datasets`: HuggingFace Datasets
- `sentence-transformers`: For independence scoring
- `bitsandbytes`: 4-bit quantization
- `wandb`: Experiment tracking

## Hardware Requirements

- **Training**: 24GB+ VRAM (RTX 3090/4090, A100)
- **Inference**: 16GB+ VRAM with 4-bit quantization

## Output Artifacts

After training:
```
grpo_planner_adapter/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
├── training_config.json
└── checkpoint-*/
```

## W&B Logging

Training logs include:
- Loss curves
- Reward distributions
- Correctness rate over time
- Decomposition rate
- Token efficiency metrics
- Per-batch example tables

## Notes

- Use gradient checkpointing for memory efficiency
- 4-bit quantization reduces VRAM requirements significantly
- Monitor W&B for training progress and debugging
- Save checkpoints regularly for fault tolerance
