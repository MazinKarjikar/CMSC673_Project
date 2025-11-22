"""
Compare Base Model vs Trained Model on GSM8K

Evaluates both:
1. Base model (meta-llama/Llama-3.1-8B-Instruct) - direct solving
2. Trained model (with planner LoRA) - using recursive decomposition pipeline

Usage:
    python gsm8k_compare.py --samples 20
"""

import os
import json
import re
import argparse
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from datasets import load_dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ComparisonConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    base_model: str = None  # Auto-detected
    planner_adapter: str = "planner"
    max_tokens: int = 512
    temperature: float = 0.7


# =============================================================================
# JSON PARSING (matches training)
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


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K solution."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def check_correct(predicted, ground_truth) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_num = extract_number(str(predicted)) if predicted else None
    gt_num = extract_number(str(ground_truth)) if ground_truth else None
    
    if pred_num is None or gt_num is None:
        return False
    
    return abs(pred_num - gt_num) < 0.01


# =============================================================================
# MODEL EVALUATORS
# =============================================================================

class BaseModelEvaluator:
    """Evaluates the base model with direct solving (no decomposition)."""
    
    def __init__(self, client: OpenAI, model_name: str, config: ComparisonConfig):
        self.client = client
        self.model_name = model_name
        self.config = config
    
    def solve(self, question: str) -> Tuple[Optional[float], str]:
        """Solve problem directly with base model."""
        prompt = f"""Solve this math problem step by step:

Problem: {question}

Show your work clearly and provide the final numerical answer.

Return JSON: {{"solution": "your step-by-step solution", "final_answer": <number>}}"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        output = response.choices[0].message.content
        parsed = parse_json_response(output)
        
        if parsed and 'final_answer' in parsed:
            return parsed['final_answer'], output
        
        return extract_number(output), output


class TrainedModelEvaluator:
    """Evaluates the trained model using the full pipeline (planner + worker)."""
    
    def __init__(self, client: OpenAI, base_model: str, planner_adapter: str, config: ComparisonConfig):
        self.client = client
        self.base_model = base_model
        self.planner_adapter = planner_adapter
        self.config = config
    
    def _generate(self, prompt: str, use_planner: bool = False) -> str:
        """Generate response using base model or planner adapter."""
        model = self.planner_adapter if use_planner else self.base_model
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        return response.choices[0].message.content
    
    def _plan(self, question: str) -> Dict:
        """Use planner (with LoRA) to analyze problem."""
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
        
        output = self._generate(prompt, use_planner=True)
        parsed = parse_json_response(output)
        
        if parsed:
            return {
                'should_decompose': parsed.get('should_decompose', False),
                'subproblems': parsed.get('subproblems', []),
                'plan': parsed.get('plan', '')
            }
        
        return {'should_decompose': False, 'subproblems': [], 'plan': 'fallback'}
    
    def _solve_subproblem(self, subproblem: str, original_question: str) -> str:
        """Worker solves a single subproblem."""
        prompt = f"""Solve this specific part of a math problem:

Original question: {original_question}
Subproblem to solve: {subproblem}

Solve ONLY this subproblem. Show your calculation and give a numerical result.

Return JSON: {{"solution": "your calculation", "result": <number>}}"""
        
        return self._generate(prompt, use_planner=False)
    
    def _combine_solutions(self, original_question: str, subproblems: List[str], solutions: List[str]) -> str:
        """Worker combines solutions."""
        solutions_text = ""
        for i, (sp, sol) in enumerate(zip(subproblems, solutions)):
            solutions_text += f"Step {i+1}: {sp}\nResult: {sol}\n\n"
        
        prompt = f"""Combine these partial solutions to answer the original question:

Original question: {original_question}

Partial solutions:
{solutions_text}

Use these results to compute the final answer. Show your work.

Return JSON: {{"final_calculation": "how you combined results", "final_answer": <number>}}"""
        
        return self._generate(prompt, use_planner=False)
    
    def _solve_directly(self, question: str) -> str:
        """Worker solves directly."""
        prompt = f"""Solve this math problem step by step:

Problem: {question}

Show your work clearly and provide the final numerical answer.

Return JSON: {{"solution": "your step-by-step solution", "final_answer": <number>}}"""
        
        return self._generate(prompt, use_planner=False)
    
    def solve(self, question: str) -> Tuple[Optional[float], str, Dict]:
        """Solve using full pipeline: planner -> worker."""
        plan_result = self._plan(question)
        
        if plan_result['should_decompose'] and plan_result['subproblems']:
            # Decomposed path
            solutions = []
            for sp in plan_result['subproblems']:
                sol = self._solve_subproblem(sp, question)
                solutions.append(sol)
            
            combined = self._combine_solutions(question, plan_result['subproblems'], solutions)
            parsed = parse_json_response(combined)
            
            if parsed and 'final_answer' in parsed:
                answer = parsed['final_answer']
            else:
                answer = extract_number(combined)
            
            return answer, combined, {
                'decomposed': True,
                'num_subproblems': len(plan_result['subproblems']),
                'plan': plan_result['plan']
            }
        else:
            # Atomic path
            response = self._solve_directly(question)
            parsed = parse_json_response(response)
            
            if parsed and 'final_answer' in parsed:
                answer = parsed['final_answer']
            else:
                answer = extract_number(response)
            
            return answer, response, {
                'decomposed': False,
                'num_subproblems': 0,
                'plan': plan_result['plan']
            }


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

def run_comparison(config: ComparisonConfig, num_samples: int = 20, verbose: bool = True):
    """Run comparison between base model and trained model."""
    
    print("\n" + "="*70)
    print("GSM8K COMPARISON: Base Model vs Trained Model (with Planner LoRA)")
    print("="*70)
    
    # Initialize client
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    
    # Get available models
    models = client.models.list()
    available = [m.id for m in models.data]
    print(f"\nAvailable models: {available}")
    
    # Find base model
    base_model = config.base_model
    if base_model is None:
        for m in available:
            if config.planner_adapter not in m:
                base_model = m
                break
    
    print(f"Base model: {base_model}")
    print(f"Planner adapter: {config.planner_adapter}")
    
    # Create evaluators
    base_evaluator = BaseModelEvaluator(client, base_model, config)
    trained_evaluator = TrainedModelEvaluator(client, base_model, config.planner_adapter, config)
    
    # Load dataset
    print(f"\nLoading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    # Results tracking
    base_results = {'correct': 0, 'total': 0, 'errors': 0}
    trained_results = {'correct': 0, 'total': 0, 'errors': 0, 'decomposed': 0}
    detailed_results = []
    
    print(f"\nEvaluating {num_samples} samples...\n")
    print("-"*70)
    
    for i, item in enumerate(dataset['test']):
        if i >= num_samples:
            break
        
        question = item['question']
        ground_truth = extract_gsm8k_answer(item['answer'])
        
        if ground_truth is None:
            continue
        
        gt_num = float(ground_truth)
        
        if verbose:
            print(f"\n[{i+1}/{num_samples}] {question[:60]}...")
            print(f"  Ground Truth: {gt_num}")
        
        result_entry = {
            'question': question,
            'ground_truth': gt_num,
        }
        
        # Evaluate base model
        try:
            base_answer, base_output = base_evaluator.solve(question)
            base_correct = check_correct(base_answer, gt_num)
            base_results['total'] += 1
            if base_correct:
                base_results['correct'] += 1
            
            result_entry['base_answer'] = base_answer
            result_entry['base_correct'] = base_correct
            
            if verbose:
                status = "✓" if base_correct else "✗"
                print(f"  Base Model:    {status} {base_answer}")
        except Exception as e:
            base_results['errors'] += 1
            result_entry['base_answer'] = None
            result_entry['base_correct'] = False
            if verbose:
                print(f"  Base Model:    ERROR - {e}")
        
        # Evaluate trained model
        try:
            trained_answer, trained_output, meta = trained_evaluator.solve(question)
            trained_correct = check_correct(trained_answer, gt_num)
            trained_results['total'] += 1
            if trained_correct:
                trained_results['correct'] += 1
            if meta['decomposed']:
                trained_results['decomposed'] += 1
            
            result_entry['trained_answer'] = trained_answer
            result_entry['trained_correct'] = trained_correct
            result_entry['decomposed'] = meta['decomposed']
            result_entry['num_subproblems'] = meta['num_subproblems']
            
            if verbose:
                status = "✓" if trained_correct else "✗"
                mode = f"(D:{meta['num_subproblems']})" if meta['decomposed'] else "(A)"
                print(f"  Trained Model: {status} {trained_answer} {mode}")
        except Exception as e:
            trained_results['errors'] += 1
            result_entry['trained_answer'] = None
            result_entry['trained_correct'] = False
            if verbose:
                print(f"  Trained Model: ERROR - {e}")
        
        detailed_results.append(result_entry)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    base_acc = 100 * base_results['correct'] / max(1, base_results['total'])
    trained_acc = 100 * trained_results['correct'] / max(1, trained_results['total'])
    decompose_rate = 100 * trained_results['decomposed'] / max(1, trained_results['total'])
    
    print(f"\n{'Metric':<30} {'Base Model':<20} {'Trained Model':<20}")
    print("-"*70)
    print(f"{'Correct':<30} {base_results['correct']:<20} {trained_results['correct']:<20}")
    print(f"{'Total':<30} {base_results['total']:<20} {trained_results['total']:<20}")
    print(f"{'Accuracy':<30} {base_acc:.1f}%{'':<17} {trained_acc:.1f}%")
    print(f"{'Errors':<30} {base_results['errors']:<20} {trained_results['errors']:<20}")
    print(f"{'Decomposed':<30} {'N/A':<20} {trained_results['decomposed']} ({decompose_rate:.1f}%)")
    
    # Improvement analysis
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)
    
    improvement = trained_acc - base_acc
    if improvement > 0:
        print(f"✓ Trained model is {improvement:.1f}% better than base model")
    elif improvement < 0:
        print(f"✗ Base model is {-improvement:.1f}% better than trained model")
    else:
        print("= Both models have the same accuracy")
    
    # Count where each model won
    base_only_correct = sum(1 for r in detailed_results 
                           if r.get('base_correct') and not r.get('trained_correct'))
    trained_only_correct = sum(1 for r in detailed_results 
                               if r.get('trained_correct') and not r.get('base_correct'))
    both_correct = sum(1 for r in detailed_results 
                       if r.get('base_correct') and r.get('trained_correct'))
    both_wrong = sum(1 for r in detailed_results 
                     if not r.get('base_correct') and not r.get('trained_correct'))
    
    print(f"\nPer-question breakdown:")
    print(f"  Both correct:         {both_correct}")
    print(f"  Both wrong:           {both_wrong}")
    print(f"  Only base correct:    {base_only_correct}")
    print(f"  Only trained correct: {trained_only_correct}")
    
    # Save results
    results_file = "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'base_accuracy': base_acc,
                'trained_accuracy': trained_acc,
                'base_correct': base_results['correct'],
                'trained_correct': trained_results['correct'],
                'total': base_results['total'],
                'decompose_rate': decompose_rate,
            },
            'detailed': detailed_results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    print("\n" + "="*70)
    
    return base_results, trained_results, detailed_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare Base vs Trained Model on GSM8K")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--planner-adapter", type=str, default="planner")
    parser.add_argument("--samples", type=int, default=20, help="Number of GSM8K samples")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    config = ComparisonConfig(
        base_url=args.base_url,
        planner_adapter=args.planner_adapter,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    run_comparison(config, num_samples=args.samples, verbose=not args.quiet)


if __name__ == "__main__":
    main()

