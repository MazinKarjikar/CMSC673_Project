"""
GRPO Training for Recursive Reasoner - Planner Only

Trains ONLY the planner model to generate good subproblem decompositions.
The worker model remains frozen and is used to execute the pipeline:

Pipeline (one-step recursion):
1. Planner generates subproblems for a question
2. Worker (frozen) solves each subproblem  
3. Worker (frozen) combines solutions into final answer
4. Reward = f(correctness, token_efficiency, subproblem_independence)

Only the planner LoRA weights are saved.
"""

import os
import re
import json
import torch
import gc
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import wandb


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass 
class TrainingArgs:
    """Training configuration."""
    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # LoRA (only for planner)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # GRPO
    num_generations: int = 4
    max_completion_length: int = 512
    max_prompt_length: int = 1024
    temperature: float = 0.7
    
    # Training
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    max_steps: int = 500
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100
    
    # Dataset
    dataset_name: str = "openai/gsm8k"
    max_samples: int = 1000
    
    # Output
    output_dir: str = "./grpo_planner_adapter"
    
    # Reward weights
    correctness_weight: float = 1.0      # Main reward: did we get correct answer?
    token_efficiency_weight: float = 0.2  # Penalty for long solutions
    independence_weight: float = 0.3      # Reward for independent subproblems
    
    # Wandb
    wandb_project: str = "recursive-reasoner-grpo"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True
    
    # Device
    use_4bit: bool = True
    
    # Worker config
    worker_max_tokens: int = 256  # Max tokens for worker responses


# =============================================================================
# WORKER MODEL (FROZEN)
# =============================================================================

class FrozenWorker:
    """
    Frozen worker model that solves subproblems and combines solutions.
    Used during reward computation - not trained.
    """
    
    def __init__(self, model, tokenizer, max_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.model.eval()  # Ensure eval mode
        
    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[str, int]:
        """Generate response and return (response, token_count)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        token_count = len(response_tokens)
        
        return response, token_count
    
    def solve_subproblem(self, subproblem: str, original_question: str) -> Tuple[str, int]:
        """Solve a single subproblem."""
        prompt = f"""Solve this specific part of a math problem:

Original question: {original_question}
Subproblem to solve: {subproblem}

Solve ONLY this subproblem. Show your calculation and give a numerical result.

Return JSON: {{"solution": "your calculation", "result": <number>}}"""
        
        return self.generate(prompt)
    
    def combine_solutions(
        self, 
        original_question: str,
        subproblems: List[str],
        solutions: List[str]
    ) -> Tuple[str, int]:
        """Combine subproblem solutions into final answer."""
        solutions_text = ""
        for i, (sp, sol) in enumerate(zip(subproblems, solutions)):
            solutions_text += f"Step {i+1}: {sp}\nResult: {sol}\n\n"
        
        prompt = f"""Combine these partial solutions to answer the original question:

Original question: {original_question}

Partial solutions:
{solutions_text}

Use these results to compute the final answer. Show your work.

Return JSON: {{"final_calculation": "how you combined results", "final_answer": <number>}}"""
        
        return self.generate(prompt)
    
    def solve_directly(self, question: str) -> Tuple[str, int]:
        """Solve a question directly without decomposition."""
        prompt = f"""Solve this math problem step by step:

Problem: {question}

Show your work clearly and provide the final numerical answer.

Return JSON: {{"solution": "your step-by-step solution", "final_answer": <number>}}"""
        
        return self.generate(prompt)


# =============================================================================
# PIPELINE EXECUTOR
# =============================================================================

class PipelineExecutor:
    """
    Executes the full pipeline: planner output -> worker execution -> final answer.
    Used for computing rewards during training.
    """
    
    def __init__(self, worker: FrozenWorker, tokenizer):
        self.worker = worker
        self.tokenizer = tokenizer
        self.similarity_model = None  # Lazy load
        
    def _load_similarity_model(self):
        if self.similarity_model is None:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def parse_planner_output(self, output: str) -> Optional[Dict]:
        """Parse planner JSON output - finds first valid JSON with expected keys."""
        if not output:
            return None
        
        # Clean up code blocks
        output = re.sub(r"```(?:json|python)?", "", output)
        output = re.sub(r"```", "", output)
        
        # Find all potential JSON objects by tracking brace depth
        # This handles the case where the model outputs multiple JSON blocks
        json_candidates = []
        depth = 0
        start = None
        
        for i, char in enumerate(output):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    json_candidates.append(output[start:i+1])
                    start = None
        
        # Try each candidate, prefer ones with "should_decompose" key
        for candidate in json_candidates:
            try:
                # Fix newlines in strings
                candidate = re.sub(
                    r'"([^"]*(?:\\"[^"]*)*)"',
                    lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
                    candidate
                )
                parsed = json.loads(candidate)
                
                # Return first valid JSON with expected key
                if isinstance(parsed, dict) and "should_decompose" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Fallback: try to find any valid JSON
        for candidate in json_candidates:
            try:
                candidate = re.sub(
                    r'"([^"]*(?:\\"[^"]*)*)"',
                    lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
                    candidate
                )
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except:
                continue
        
        return None
    
    def extract_number(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        if text is None:
            return None
        try:
            return float(str(text).replace(',', ''))
        except:
            pass
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', str(text))
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except:
                pass
        return None
    
    def compute_independence(self, subproblems: List[str]) -> float:
        """Compute independence score for subproblems (higher = more independent)."""
        if not subproblems or len(subproblems) < 2:
            return 0.5  # Neutral for single/no subproblems
        
        self._load_similarity_model()
        embeddings = self.similarity_model.encode(subproblems)
        sims = cosine_similarity(embeddings)
        
        n = len(subproblems)
        avg_sim = sum(sims[i][j] for i in range(n) for j in range(i+1, n)) / max(1, n*(n-1)/2)
        
        # Independence = 1 - similarity
        return 1.0 - avg_sim
    
    def execute_pipeline(
        self, 
        planner_output: str, 
        original_question: str,
        ground_truth: str
    ) -> Dict:
        """
        Execute the full pipeline and compute metrics.
        
        Returns:
            Dict with:
            - predicted_answer: the final answer
            - is_correct: bool
            - total_tokens: sum of tokens across all worker calls
            - independence_score: how independent the subproblems are
            - num_subproblems: number of subproblems
            - decomposed: whether decomposition was used
        """
        parsed = self.parse_planner_output(planner_output)
        
        # Default result for failed parsing
        default_result = {
            "predicted_answer": None,
            "is_correct": False,
            "total_tokens": 0,
            "independence_score": 0.0,
            "num_subproblems": 0,
            "decomposed": False,
            "parse_failed": True
        }
        
        if parsed is None:
            return default_result
        
        should_decompose = parsed.get("should_decompose", False)
        subproblems = parsed.get("subproblems", [])
        
        total_tokens = 0
        
        if should_decompose and subproblems and len(subproblems) >= 1:
            # Execute decomposed pipeline
            solutions = []
            
            for sp in subproblems:
                sol, tokens = self.worker.solve_subproblem(sp, original_question)
                solutions.append(sol)
                total_tokens += tokens
            
            # Combine solutions
            final_response, combine_tokens = self.worker.combine_solutions(
                original_question, subproblems, solutions
            )
            total_tokens += combine_tokens
            
            # Extract final answer
            parsed_final = self.parse_planner_output(final_response)
            if parsed_final:
                predicted = parsed_final.get("final_answer")
            else:
                predicted = self.extract_number(final_response)
            
            independence = self.compute_independence(subproblems)
            
            return {
                "predicted_answer": predicted,
                "is_correct": self._check_correct(predicted, ground_truth),
                "total_tokens": total_tokens,
                "independence_score": independence,
                "num_subproblems": len(subproblems),
                "decomposed": True,
                "parse_failed": False
            }
        else:
            # Solve directly (atomic)
            response, tokens = self.worker.solve_directly(original_question)
            total_tokens = tokens
            
            parsed_response = self.parse_planner_output(response)
            if parsed_response:
                predicted = parsed_response.get("final_answer")
            else:
                predicted = self.extract_number(response)
            
            return {
                "predicted_answer": predicted,
                "is_correct": self._check_correct(predicted, ground_truth),
                "total_tokens": total_tokens,
                "independence_score": 0.5,  # Neutral for atomic
                "num_subproblems": 0,
                "decomposed": False,
                "parse_failed": False
            }
    
    def _check_correct(self, predicted, ground_truth) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_num = self.extract_number(str(predicted)) if predicted else None
        gt_num = self.extract_number(ground_truth)
        
        if pred_num is None or gt_num is None:
            return False
        
        return abs(pred_num - gt_num) < 0.01


# =============================================================================
# REWARD FUNCTION
# =============================================================================

class PipelineRewardFunction:
    """
    Computes rewards by executing the full pipeline.
    
    Reward = correctness_weight * correct 
           + token_efficiency_weight * (1 - tokens/max_tokens)
           + independence_weight * independence_score
    """
    
    # Required by TRL GRPOTrainer
    __name__ = "pipeline_reward"
    
    def __init__(
        self, 
        executor: PipelineExecutor,
        args: TrainingArgs,
        questions: List[str],
        ground_truths: List[str]
    ):
        self.executor = executor
        self.args = args
        self.questions = questions
        self.ground_truths = ground_truths
        self.max_expected_tokens = 1000  # For normalization
        self.call_count = 0  # Track calls for logging
        self.log_every_n_calls = 1  # Log every N reward computations
        
        # Create lookup table: question -> ground_truth
        # This is critical because the trainer shuffles the dataset!
        self.question_to_gt = {q: gt for q, gt in zip(questions, ground_truths)}
        
    def _extract_question_from_prompt(self, prompt: str) -> str:
        """Extract the original question from the planner prompt."""
        # The prompt format is:
        # "Analyze this math problem and decide how to solve it:\n\nProblem: {question}\n\nYour task:..."
        match = re.search(r'Problem:\s*(.+?)\n\nYour task:', prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: try to find anything after "Problem:"
        match = re.search(r'Problem:\s*(.+?)(?:\n\n|$)', prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
        
    def __call__(
        self,
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """Compute rewards for planner completions."""
        rewards = []
        self.call_count += 1
        
        # Debug: print what kwargs are available on first call
        if self.call_count == 1:
            print(f"\n  DEBUG - Reward function kwargs: {list(kwargs.keys())}")
            for key, val in kwargs.items():
                if isinstance(val, list):
                    print(f"    {key}: list of {len(val)} items")
                    if val and isinstance(val[0], str):
                        print(f"      First item preview: {val[0][:100]}...")
                else:
                    print(f"    {key}: {type(val).__name__}")
        
        # Get prompts - these are the ACTUAL prompts used for generation!
        # TRL GRPOTrainer passes prompts in kwargs
        prompts = kwargs.get("prompts", kwargs.get("prompt", []))
        
        # Prepare wandb table data
        table_data = []
        
        for i, completion in enumerate(completions):
            # Extract the original question FROM the actual prompt used
            if prompts and i < len(prompts):
                prompt = prompts[i]
                question = self._extract_question_from_prompt(prompt)
                # Look up ground truth using the question
                gt = self.question_to_gt.get(question, "")
                
                # Debug: if no match found, try fuzzy matching
                if not gt and question:
                    # Try to find a close match
                    for q, g in self.question_to_gt.items():
                        if question[:50] in q or q[:50] in question:
                            gt = g
                            question = q  # Use the full matched question
                            break
                    if not gt:
                        print(f"  WARNING: Could not find ground truth for question: {question[:100]}...")
            else:
                # Fallback to old method (shouldn't happen if prompts are passed)
                idx = i % len(self.questions)
                question = self.questions[idx]
                gt = self.ground_truths[idx]
                print(f"  WARNING: No prompt available for completion {i}, using fallback index")
            
            # Debug: on first few calls, print diagnostics
            if self.call_count <= 2 and i < 2:
                print(f"\n  DEBUG - Call {self.call_count}, Sample {i}:")
                print(f"    Question: {question[:60]}..." if question else "    Question: NONE")
                print(f"    GT: {gt}")
                print(f"    Completion preview: {completion[:150]}...")
                
                # Try parsing and show result
                parsed_test = self.executor.parse_planner_output(completion)
                if parsed_test:
                    print(f"    Parsed: decompose={parsed_test.get('should_decompose')}, "
                          f"subprobs={len(parsed_test.get('subproblems', []))}")
                else:
                    print(f"    Parsed: FAILED")
            
            # Execute pipeline
            result = self.executor.execute_pipeline(completion, question, gt)
            
            # Compute reward components
            
            # 1. Correctness (main reward)
            correctness_reward = 1.0 if result["is_correct"] else 0.0
            
            # 2. Token efficiency (lower tokens = higher reward)
            token_ratio = min(result["total_tokens"] / self.max_expected_tokens, 1.0)
            efficiency_reward = 1.0 - token_ratio
            
            # 3. Independence (higher = better, only if decomposed)
            if result["decomposed"]:
                independence_reward = result["independence_score"]
            else:
                independence_reward = 0.5  # Neutral for atomic
            
            # 4. Format penalty for parse failures
            if result.get("parse_failed", False):
                total_reward = 0.1  # Small reward for attempting
            else:
                total_reward = (
                    self.args.correctness_weight * correctness_reward +
                    self.args.token_efficiency_weight * efficiency_reward +
                    self.args.independence_weight * independence_reward
                )
            
            rewards.append(total_reward)
            
            # Collect data for wandb table
            table_data.append({
                "question": question,
                "ground_truth": gt,
                "completion": completion,
                "predicted_answer": str(result.get("predicted_answer", "N/A")),
                "is_correct": result["is_correct"],
                "decomposed": result["decomposed"],
                "num_subproblems": result["num_subproblems"],
                "total_tokens": result["total_tokens"],
                "independence_score": round(result["independence_score"], 3),
                "reward": round(total_reward, 3),
            })
            
            # Log some examples for debugging
            if i < 2:  # Log first 2 examples
                print(f"  [Sample {i}] Correct: {result['is_correct']}, "
                      f"Tokens: {result['total_tokens']}, "
                      f"Decomposed: {result['decomposed']}, "
                      f"Reward: {total_reward:.3f}")
        
        # Log to wandb
        if self.args.use_wandb and self.call_count % self.log_every_n_calls == 0:
            try:
                # Create wandb table
                columns = ["question", "ground_truth", "completion", "predicted_answer", 
                          "is_correct", "decomposed", "num_subproblems", "total_tokens",
                          "independence_score", "reward"]
                table = wandb.Table(columns=columns)
                for row in table_data:
                    table.add_data(*[row[col] for col in columns])
                
                # Log table and summary metrics
                correct_count = sum(1 for r in table_data if r["is_correct"])
                decomposed_count = sum(1 for r in table_data if r["decomposed"])
                avg_tokens = np.mean([r["total_tokens"] for r in table_data])
                avg_reward = np.mean(rewards)
                
                wandb.log({
                    "completions_table": table,
                    "batch/correct_rate": correct_count / len(table_data),
                    "batch/decompose_rate": decomposed_count / len(table_data),
                    "batch/avg_tokens": avg_tokens,
                    "batch/avg_reward": avg_reward,
                    "batch/num_samples": len(table_data),
                })
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")
        
        return rewards


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K solution."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def prepare_dataset(dataset, max_samples: int = 1000) -> Tuple[Dataset, List[str], List[str]]:
    """
    Prepare GSM8K for planner training.
    Returns dataset, questions list, and ground truths list.
    """
    prompts = []
    questions = []
    answers = []
    
    for i, item in enumerate(dataset['train']):
        if i >= max_samples:
            break
            
        question = item['question']
        answer = extract_answer(item['answer'])
        
        # Planner prompt
        prompt = f"""Analyze this math problem and decide how to solve it:

Problem: {question}

Your task:
1. Decide if this problem should be decomposed into subproblems or solved directly
2. If decomposing: identify 2-3 distinct calculation steps as subproblems
3. Explain your approach briefly

DECOMPOSE if the problem has multiple distinct calculation steps.
Keep ATOMIC if it's a simple single-step calculation.

Return ONLY valid JSON:
{{
    "should_decompose": true/false,
    "subproblems": ["step 1", "step 2", ...] or [] if atomic,
    "plan": "Brief explanation of approach"
}}"""
        
        prompts.append(prompt)
        questions.append(question)
        answers.append(answer)
    
    dataset = Dataset.from_dict({"prompt": prompts})
    
    return dataset, questions, answers


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_models(args: TrainingArgs):
    """
    Setup planner (with LoRA) and worker (frozen) models.
    
    Returns:
        planner_model: Model with LoRA for training
        worker: FrozenWorker instance
        tokenizer: Shared tokenizer
        lora_config: LoRA configuration
    """
    # Quantization config
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print("Loading base model for worker (frozen)...")
    # Worker model - frozen, no LoRA
    worker_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    worker_model.eval()
    for param in worker_model.parameters():
        param.requires_grad = False
    
    worker = FrozenWorker(worker_model, tokenizer, args.worker_max_tokens)
    
    print("Loading base model for planner (with LoRA)...")
    # Planner model - with LoRA for training
    # Need to load a separate instance for the planner
    planner_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    if args.use_4bit:
        planner_model = prepare_model_for_kbit_training(planner_model)
    
    # LoRA config for planner
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to planner
    planner_model = get_peft_model(planner_model, lora_config)
    planner_model.print_trainable_parameters()
    
    return planner_model, worker, tokenizer, lora_config


# =============================================================================
# TRAINING
# =============================================================================

def train(args: TrainingArgs):
    """Train the planner model using GRPO."""
    
    print("\n" + "="*60)
    print("GRPO Training - Planner Only (One-Step Recursion)")
    print("="*60)
    print(f"Reward weights:")
    print(f"  Correctness: {args.correctness_weight}")
    print(f"  Token efficiency: {args.token_efficiency_weight}")
    print(f"  Independence: {args.independence_weight}")
    print("="*60)
    
    # Load dataset
    print("\nLoading GSM8K dataset...")
    gsm8k = load_dataset(args.dataset_name, "main")
    dataset, questions, ground_truths = prepare_dataset(gsm8k, args.max_samples)
    print(f"Dataset size: {len(dataset)}")
    
    # Setup models
    print("\nSetting up models...")
    planner_model, worker, tokenizer, lora_config = setup_models(args)
    
    # Create pipeline executor
    executor = PipelineExecutor(worker, tokenizer)
    
    # Create reward function
    reward_fn = PipelineRewardFunction(executor, args, questions, ground_truths)
    
    # Initialize wandb
    if args.use_wandb:
        run_name = args.wandb_run_name or f"grpo-planner-{args.max_steps}steps"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "base_model": args.base_model,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_steps": args.max_steps,
                "num_generations": args.num_generations,
                "correctness_weight": args.correctness_weight,
                "token_efficiency_weight": args.token_efficiency_weight,
                "independence_weight": args.independence_weight,
            },
            reinit=True,
        )
    
    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        
        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        
        # Training
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        
        # Logging & Saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # Wandb
        report_to="wandb" if args.use_wandb else "none",
        run_name=run_name if args.use_wandb else None,
        
        # Other
        bf16=True,
        remove_unused_columns=False,
    )
    
    # Create trainer
    print("\nInitializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=planner_model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=lora_config,
    )
    
    # Train
    print("\nStarting training...")
    print("(Each step runs full pipeline: planner -> worker solves -> worker combines)")
    trainer.train()
    
    # Save
    print(f"\nSaving planner adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save config
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "correctness_weight": args.correctness_weight,
            "token_efficiency_weight": args.token_efficiency_weight,
            "independence_weight": args.independence_weight,
        }, f, indent=2)
    
    if args.use_wandb:
        wandb.finish()
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Planner adapter saved to: {args.output_dir}")
    print("="*60)
    print("\nTo use with vLLM:")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"    --model {args.base_model} \\")
    print(f"    --enable-lora \\")
    print(f"    --lora-modules planner={args.output_dir}")
    
    return trainer


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GRPO Training - Planner Only (One-Step Recursion)"
    )
    
    # Model
    parser.add_argument("--base-model", type=str, 
                       default="meta-llama/Llama-3.1-8B-Instruct")
    
    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    
    # GRPO
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    # Reward weights
    parser.add_argument("--correctness-weight", type=float, default=1.0,
                       help="Weight for correctness reward")
    parser.add_argument("--token-weight", type=float, default=0.2,
                       help="Weight for token efficiency reward")
    parser.add_argument("--independence-weight", type=float, default=0.3,
                       help="Weight for subproblem independence reward")
    
    # Dataset
    parser.add_argument("--max-samples", type=int, default=1000)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./grpo_planner_adapter")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="recursive-reasoner-grpo")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    
    # Other
    parser.add_argument("--no-4bit", action="store_true")
    
    args_cli = parser.parse_args()
    
    # Create config
    args = TrainingArgs(
        base_model=args_cli.base_model,
        lora_r=args_cli.lora_r,
        lora_alpha=args_cli.lora_alpha,
        learning_rate=args_cli.lr,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_accum,
        max_steps=args_cli.max_steps,
        num_generations=args_cli.num_generations,
        temperature=args_cli.temperature,
        correctness_weight=args_cli.correctness_weight,
        token_efficiency_weight=args_cli.token_weight,
        independence_weight=args_cli.independence_weight,
        max_samples=args_cli.max_samples,
        output_dir=args_cli.output_dir,
        wandb_project=args_cli.wandb_project,
        wandb_run_name=args_cli.wandb_run_name,
        use_wandb=not args_cli.no_wandb,
        use_4bit=not args_cli.no_4bit,
    )
    
    # Print config
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps}")
    print(f"Max steps: {args.max_steps}")
    print(f"Num generations (GRPO): {args.num_generations}")
    print(f"Dataset: GSM8K ({args.max_samples} samples)")
    print(f"Output: {args.output_dir}")
    print(f"Reward weights: correct={args.correctness_weight}, "
          f"tokens={args.token_efficiency_weight}, "
          f"independence={args.independence_weight}")
    print("="*60)
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
