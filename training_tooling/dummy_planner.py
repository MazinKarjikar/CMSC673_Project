import os
from typing import Dict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# -------------------------
# Config
# -------------------------

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "emilbiju/Planning-Data-Math-Full-Soln"
OUTPUT_DIR = "./deepseek-r1-planner-lora"

MAX_SEQ_LEN = 17000
BATCH_SIZE = 2          # per device
GRAD_ACCUM = 8          # effective batch = 16
LR = 1e-4
NUM_EPOCHS = 1.0
WARMUP_RATIO = 0.03

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Match actual module names as seen in HF dump (Qwen/DeepSeek-R1-Distill):
# model.layers.X.self_attn.{q_proj,k_proj,v_proj,o_proj}
# model.layers.X.mlp.{gate_proj,up_proj,down_proj}
LORA_TARGET_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def format_example(example: Dict) -> str:
    """
    Turn one dataset row into a single training string.

    We treat:
      - SystemPrompt: planner system description
      - UserPrompt: question + prior plans/executions
      - ExpectedOutput: the <Plan_p>...</Plan_p> block for this phase

    Modern TRL expects formatting_func to return a single string per example.
    """
    system = example["SystemPrompt"].strip()
    user = example["UserPrompt"].strip()
    target = example["ExpectedOutput"].strip()

    # This mirrors how we'll prompt the planner at inference time:
    prompt = system + "\n\n" + user + "\n"
    full_text = prompt + target
    return full_text


def main():
    # -------------------------
    # Tokenizer & model
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Right-side padding is standard for causal LM training
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",           # modern transformers arg
        device_map="auto",
        trust_remote_code=True,
    )

    # Needed when using gradient checkpointing
    model.config.use_cache = False

    # -------------------------
    # Dataset
    # -------------------------
    dataset = load_dataset(DATASET_NAME)

    if "train" in dataset:
        train_ds = dataset["train"]
        # Prefer "validation" if it exists, else fall back to "test" if available
        if "validation" in dataset:
            val_ds = dataset["validation"]
        elif "test" in dataset:
            val_ds = dataset["test"]
        else:
            val_ds = None
    else:
        # fallback: treat first split as full, make tiny val
        first_split_name = list(dataset.keys())[0]
        full = dataset[first_split_name]
        split = full.train_test_split(test_size=0.02, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    # OPTIONAL: partial fine-tune (uncomment)
    # train_ds = train_ds.select(range(5000))

    # -------------------------
    # LoRA config
    # -------------------------
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -------------------------
    # Training args (modern: SFTConfig)
    # -------------------------
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        max_steps=5,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=200 if val_ds is not None else None,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",

        # TRL-specific config bits now live here:
        max_length=MAX_SEQ_LEN,   # replaces old max_seq_length
        packing=False,            # explicit, even though it's the default
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,   # modern name for tokenizer arg
        formatting_func=format_example,
    )

    trainer.train()

    # For a PEFT model, this saves just the adapter weights
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved planner LoRA to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()