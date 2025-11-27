#!/usr/bin/env python3
"""
Three-Way Benchmark: Original vs Base Recursive vs Trained Recursive

Compares three configurations across multiple math benchmarks:
  A) Original Model - Direct solving with base model
  B) Base Recursive - Base model as both planner and worker  
  C) Trained Recursive - Trained LoRA planner + base model worker

Supported benchmarks:
  - GSM8K: Grade school math (8.5K problems)
  - MATH: Competition math (12.5K problems)
  - AIME: American Invitational Mathematics Examination
  - AQuA: Algebraic word problems

Usage:
    # Single server with LoRA (recommended)
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --enable-lora \
        --lora-modules planner=./grpo_planner_adapter \
        --max-lora-rank 64 \
        --port 8000

    # Run benchmarks
    python benchmark.py --datasets gsm8k math --samples 50
    python benchmark.py --datasets all --samples 100
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    base_model: str = None  # Auto-detected
    planner_adapter: str = "planner"
    max_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.9


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""
    dataset: str
    config_name: str  # "original", "base_recursive", "trained_recursive"
    correct: int = 0
    total: int = 0
    errors: int = 0
    decomposed: int = 0
    total_time: float = 0.0
    detailed: List[Dict] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        return 100 * self.correct / max(1, self.total)
    
    @property
    def decompose_rate(self) -> float:
        return 100 * self.decomposed / max(1, self.total)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / max(1, self.total)


# =============================================================================
# JSON PARSING
# =============================================================================

def parse_json_response(output: str) -> Optional[Dict]:
    """Parse JSON from model output using brace depth tracking."""
    if not output:
        return None
    
    # Clean up code blocks
    output = re.sub(r"```(?:json|python)?", "", output)
    output = re.sub(r"```", "", output)
    
    # Find all potential JSON objects by tracking brace depth
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
    
    # Try each candidate
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
        except json.JSONDecodeError:
            continue
    
    return None


def extract_number(text: str) -> Optional[float]:
    """Extract numerical answer from text."""
    if text is None:
        return None
    try:
        return float(str(text).replace(',', '').replace('$', '').strip())
    except:
        pass
    # Look for numbers, prefer last one
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', str(text))
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    return None


def check_correct(predicted, ground_truth, tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_num = extract_number(str(predicted)) if predicted else None
    gt_num = extract_number(str(ground_truth)) if ground_truth else None
    
    if pred_num is None or gt_num is None:
        return False
    
    # For integers, exact match; for floats, tolerance
    if gt_num == int(gt_num):
        return abs(pred_num - gt_num) < 0.5
    return abs(pred_num - gt_num) < tolerance * max(1, abs(gt_num))


# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_gsm8k(split: str = "test", max_samples: int = None) -> List[Dict]:
    """Load GSM8K dataset."""
    dataset = load_dataset("openai/gsm8k", "main")
    samples = []
    
    for item in dataset[split]:
        # Extract answer from #### format
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', item['answer'])
        if match:
            answer = match.group(1).replace(',', '')
            samples.append({
                'question': item['question'],
                'answer': answer,
                'solution': item['answer'],
            })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples


def load_math_dataset(split: str = "test", max_samples: int = None) -> List[Dict]:
    """Load MATH dataset (competition mathematics)."""
    try:
        dataset = load_dataset("hendrycks/competition_math")
    except:
        print("Warning: MATH dataset not available, skipping...")
        return []
    
    samples = []
    
    for item in dataset[split]:
        # Extract boxed answer
        solution = item.get('solution', '')
        match = re.search(r'\\boxed{([^}]+)}', solution)
        if match:
            answer = match.group(1)
            # Try to extract number if it's numeric
            num = extract_number(answer)
            if num is not None:
                samples.append({
                    'question': item['problem'],
                    'answer': str(num),
                    'solution': solution,
                    'level': item.get('level', 'Unknown'),
                    'type': item.get('type', 'Unknown'),
                })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples


def load_aqua_dataset(split: str = "test", max_samples: int = None) -> List[Dict]:
    """Load AQuA-RAT dataset (algebraic word problems with rationales)."""
    try:
        dataset = load_dataset("deepmind/aqua_rat")
    except:
        print("Warning: AQuA dataset not available, skipping...")
        return []
    
    samples = []
    
    for item in dataset[split]:
        # AQuA has multiple choice, extract the correct option
        options = item.get('options', [])
        correct = item.get('correct', '')
        
        # Find the correct answer value
        answer = None
        for opt in options:
            if opt.startswith(correct + ')'):
                # Extract number from option like "A)123" or "B)45.6"
                num = extract_number(opt)
                if num is not None:
                    answer = str(num)
                break
        
        if answer:
            samples.append({
                'question': item['question'] + "\n\nOptions: " + ", ".join(options),
                'answer': answer,
                'rationale': item.get('rationale', ''),
            })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples


def load_aime_dataset(max_samples: int = None) -> List[Dict]:
    """Load AIME problems (American Invitational Mathematics Examination)."""
    # AIME problems - these are from various public sources
    # AIME answers are always integers from 0-999
    try:
        # Try to load from HuggingFace
        dataset = load_dataset("qq8933/AIME_1983_2024")
        samples = []
        
        for item in dataset['train']:
            question = item.get('Question', item.get('problem', ''))
            answer = item.get('Answer', item.get('answer', ''))
            
            if question and answer:
                num = extract_number(str(answer))
                if num is not None:
                    samples.append({
                        'question': question,
                        'answer': str(int(num)),
                    })
            
            if max_samples and len(samples) >= max_samples:
                break
        
        return samples
    except Exception as e:
        print(f"Warning: AIME dataset not available ({e}), skipping...")
        return []


def load_dataset_by_name(name: str, max_samples: int = None) -> List[Dict]:
    """Load dataset by name."""
    loaders = {
        'gsm8k': lambda: load_gsm8k(max_samples=max_samples),
        'math': lambda: load_math_dataset(max_samples=max_samples),
        'aqua': lambda: load_aqua_dataset(max_samples=max_samples),
        'aime': lambda: load_aime_dataset(max_samples=max_samples),
    }
    
    if name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    
    return loaders[name.lower()]()


# =============================================================================
# SOLVERS
# =============================================================================

class BaseSolver:
    """Base class for all solvers."""
    
    def __init__(self, client: OpenAI, config: BenchmarkConfig):
        self.client = client
        self.config = config
        self.base_model = config.base_model
    
    def _generate(self, prompt: str, model: str = None) -> str:
        """Generate response from model."""
        model = model or self.base_model
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        return response.choices[0].message.content


class OriginalSolver(BaseSolver):
    """Solver A: Direct solving with original model (no decomposition)."""
    
    name = "original"
    
    def solve(self, question: str) -> Tuple[Optional[float], Dict]:
        """Solve problem directly."""
        prompt = f"""Solve this math problem step by step:

Problem: {question}

Show your work clearly and provide the final numerical answer.

Return JSON: {{"solution": "your step-by-step solution", "final_answer": <number>}}"""
        
        output = self._generate(prompt)
        parsed = parse_json_response(output)
        
        if parsed and 'final_answer' in parsed:
            answer = parsed['final_answer']
        else:
            answer = extract_number(output)
        
        return answer, {'decomposed': False, 'output': output}


class BaseRecursiveSolver(BaseSolver):
    """Solver B: Base model as both planner and worker."""
    
    name = "base_recursive"
    
    def _plan(self, question: str) -> Dict:
        """Plan using base model."""
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
        
        output = self._generate(prompt)
        parsed = parse_json_response(output)
        
        if parsed:
            return {
                'should_decompose': parsed.get('should_decompose', False),
                'subproblems': parsed.get('subproblems', []),
                'plan': parsed.get('plan', '')
            }
        
        return {'should_decompose': False, 'subproblems': [], 'plan': 'fallback'}
    
    def _solve_subproblem(self, subproblem: str, original: str) -> str:
        """Solve a subproblem."""
        prompt = f"""Solve this specific part of a math problem:

Original question: {original}
Subproblem to solve: {subproblem}

Solve ONLY this subproblem. Show your calculation and give a numerical result.

Return JSON: {{"solution": "your calculation", "result": <number>}}"""
        
        return self._generate(prompt)
    
    def _combine(self, original: str, subproblems: List[str], solutions: List[str]) -> str:
        """Combine solutions."""
        solutions_text = ""
        for i, (sp, sol) in enumerate(zip(subproblems, solutions)):
            parsed = parse_json_response(sol)
            result = parsed.get('result', sol) if parsed else sol
            solutions_text += f"Step {i+1}: {sp}\nResult: {result}\n\n"
        
        prompt = f"""Combine these partial solutions to answer the original question:

Original question: {original}

Partial solutions:
{solutions_text}

Use these results to compute the final answer. Show your work.

Return JSON: {{"final_calculation": "how you combined results", "final_answer": <number>}}"""
        
        return self._generate(prompt)
    
    def _solve_directly(self, question: str) -> str:
        """Solve directly (atomic path)."""
        prompt = f"""Solve this math problem step by step:

Problem: {question}

Show your work clearly and provide the final numerical answer.

Return JSON: {{"solution": "your step-by-step solution", "final_answer": <number>}}"""
        
        return self._generate(prompt)
    
    def solve(self, question: str) -> Tuple[Optional[float], Dict]:
        """Solve using recursive decomposition with base model."""
        plan = self._plan(question)
        
        if plan['should_decompose'] and plan['subproblems']:
            # Decomposed path
            solutions = []
            for sp in plan['subproblems']:
                sol = self._solve_subproblem(sp, question)
                solutions.append(sol)
            
            combined = self._combine(question, plan['subproblems'], solutions)
            parsed = parse_json_response(combined)
            
            if parsed and 'final_answer' in parsed:
                answer = parsed['final_answer']
            else:
                answer = extract_number(combined)
            
            return answer, {
                'decomposed': True,
                'num_subproblems': len(plan['subproblems']),
                'plan': plan['plan'],
                'output': combined
            }
        else:
            # Atomic path
            output = self._solve_directly(question)
            parsed = parse_json_response(output)
            
            if parsed and 'final_answer' in parsed:
                answer = parsed['final_answer']
            else:
                answer = extract_number(output)
            
            return answer, {
                'decomposed': False,
                'num_subproblems': 0,
                'plan': plan['plan'],
                'output': output
            }


class TrainedRecursiveSolver(BaseRecursiveSolver):
    """Solver C: Trained LoRA planner + base model worker."""
    
    name = "trained_recursive"
    
    def _plan(self, question: str) -> Dict:
        """Plan using trained LoRA adapter."""
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
        
        # Use trained planner (LoRA adapter)
        output = self._generate(prompt, model=self.config.planner_adapter)
        parsed = parse_json_response(output)
        
        if parsed:
            return {
                'should_decompose': parsed.get('should_decompose', False),
                'subproblems': parsed.get('subproblems', []),
                'plan': parsed.get('plan', '')
            }
        
        return {'should_decompose': False, 'subproblems': [], 'plan': 'fallback'}


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(
    solver,
    samples: List[Dict],
    dataset_name: str,
    verbose: bool = True
) -> BenchmarkResult:
    """Run benchmark for a single solver on a dataset."""
    
    result = BenchmarkResult(dataset=dataset_name, config_name=solver.name)
    
    for i, sample in enumerate(samples):
        question = sample['question']
        ground_truth = sample['answer']
        
        if verbose:
            print(f"  [{i+1}/{len(samples)}] ", end="", flush=True)
        
        start_time = time.time()
        
        try:
            answer, meta = solver.solve(question)
            elapsed = time.time() - start_time
            
            correct = check_correct(answer, ground_truth)
            
            result.total += 1
            result.total_time += elapsed
            if correct:
                result.correct += 1
            if meta.get('decomposed', False):
                result.decomposed += 1
            
            result.detailed.append({
                'question': question[:100] + '...' if len(question) > 100 else question,
                'ground_truth': ground_truth,
                'predicted': answer,
                'correct': correct,
                'decomposed': meta.get('decomposed', False),
                'time': elapsed,
            })
            
            if verbose:
                status = "✓" if correct else "✗"
                mode = "(D)" if meta.get('decomposed') else "(A)"
                print(f"{status} {mode} [{elapsed:.1f}s]")
                
        except Exception as e:
            result.errors += 1
            result.detailed.append({
                'question': question[:100],
                'ground_truth': ground_truth,
                'error': str(e),
            })
            if verbose:
                print(f"ERROR: {e}")
    
    return result


def run_all_benchmarks(
    config: BenchmarkConfig,
    datasets: List[str],
    samples_per_dataset: int,
    verbose: bool = True,
    output_dir: str = "benchmark_results"
) -> Dict[str, Dict[str, BenchmarkResult]]:
    """Run all benchmarks across all configurations."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize client
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    
    # Get available models
    models = client.models.list()
    available = [m.id for m in models.data]
    print(f"\nAvailable models: {available}")
    
    # Auto-detect base model
    if config.base_model is None:
        for m in available:
            if config.planner_adapter not in m:
                config.base_model = m
                break
    
    print(f"Base model: {config.base_model}")
    print(f"Planner adapter: {config.planner_adapter}")
    
    # Check if planner adapter is available
    has_planner = config.planner_adapter in available
    if not has_planner:
        print(f"\n⚠️  Planner adapter '{config.planner_adapter}' not found!")
        print("   Will skip trained_recursive benchmark.")
        print("   To enable, start vLLM with --enable-lora --lora-modules planner=./adapter")
    
    # Create solvers
    solvers = [
        OriginalSolver(client, config),
        BaseRecursiveSolver(client, config),
    ]
    if has_planner:
        solvers.append(TrainedRecursiveSolver(client, config))
    
    # Results storage
    all_results: Dict[str, Dict[str, BenchmarkResult]] = {}
    
    print("\n" + "="*80)
    print("THREE-WAY BENCHMARK: Original vs Base Recursive vs Trained Recursive")
    print("="*80)
    
    for dataset_name in datasets:
        print(f"\n{'─'*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'─'*80}")
        
        # Load dataset
        print(f"Loading {dataset_name}...")
        samples = load_dataset_by_name(dataset_name, max_samples=samples_per_dataset)
        
        if not samples:
            print(f"  No samples loaded, skipping...")
            continue
        
        print(f"Loaded {len(samples)} samples")
        
        all_results[dataset_name] = {}
        
        for solver in solvers:
            print(f"\n▶ {solver.name.upper().replace('_', ' ')}")
            result = run_benchmark(solver, samples, dataset_name, verbose=verbose)
            all_results[dataset_name][solver.name] = result
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Per-dataset summary
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"{'Config':<25} {'Accuracy':<12} {'Correct':<10} {'Decomp%':<10} {'Avg Time':<10}")
        print("-"*70)
        
        for config_name, result in results.items():
            print(f"{config_name:<25} {result.accuracy:>6.1f}%     "
                  f"{result.correct}/{result.total:<6} "
                  f"{result.decompose_rate:>5.1f}%     "
                  f"{result.avg_time:>6.2f}s")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL COMPARISON")
    print("="*80)
    
    overall = {name: {'correct': 0, 'total': 0, 'decomposed': 0, 'time': 0.0} 
               for name in ['original', 'base_recursive', 'trained_recursive']}
    
    for dataset_name, results in all_results.items():
        for config_name, result in results.items():
            overall[config_name]['correct'] += result.correct
            overall[config_name]['total'] += result.total
            overall[config_name]['decomposed'] += result.decomposed
            overall[config_name]['time'] += result.total_time
    
    print(f"\n{'Config':<25} {'Overall Accuracy':<20} {'Decomposition Rate':<20}")
    print("-"*70)
    
    for config_name, stats in overall.items():
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            decomp = 100 * stats['decomposed'] / stats['total']
            print(f"{config_name:<25} {acc:>6.1f}% ({stats['correct']}/{stats['total']})   {decomp:>6.1f}%")
    
    # Analysis
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)
    
    if 'original' in overall and 'trained_recursive' in overall:
        orig = overall['original']
        trained = overall['trained_recursive']
        if orig['total'] > 0 and trained['total'] > 0:
            orig_acc = 100 * orig['correct'] / orig['total']
            trained_acc = 100 * trained['correct'] / trained['total']
            diff = trained_acc - orig_acc
            
            if diff > 0:
                print(f"✓ Trained recursive is {diff:.1f}% better than original")
            elif diff < 0:
                print(f"✗ Original is {-diff:.1f}% better than trained recursive")
            else:
                print("= Both have the same accuracy")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    
    # Convert to serializable format
    serializable = {}
    for dataset_name, results in all_results.items():
        serializable[dataset_name] = {}
        for config_name, result in results.items():
            serializable[dataset_name][config_name] = {
                'accuracy': result.accuracy,
                'correct': result.correct,
                'total': result.total,
                'errors': result.errors,
                'decomposed': result.decomposed,
                'decompose_rate': result.decompose_rate,
                'total_time': result.total_time,
                'avg_time': result.avg_time,
                'detailed': result.detailed,
            }
    
    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Three-way benchmark: Original vs Base Recursive vs Trained Recursive"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k"],
        choices=["gsm8k", "math", "aqua", "aime", "all"],
        help="Datasets to benchmark"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL"
    )
    parser.add_argument(
        "--planner-adapter",
        type=str,
        default="planner",
        help="Name of planner LoRA adapter"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle "all" datasets
    if "all" in args.datasets:
        datasets = ["gsm8k", "math", "aqua", "aime"]
    else:
        datasets = args.datasets
    
    config = BenchmarkConfig(
        base_url=args.base_url,
        planner_adapter=args.planner_adapter,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    run_all_benchmarks(
        config=config,
        datasets=datasets,
        samples_per_dataset=args.samples,
        verbose=not args.quiet,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

