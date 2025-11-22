"""
Inference script using vLLM with GRPO-trained LoRA adapters.

Two ways to use:
1. Start vLLM server with LoRA adapter, then use this script
2. Use vLLM directly in Python with LoRA adapter

vLLM Server Usage:
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --enable-lora \
        --lora-modules recursive_reasoner=./grpo_lora_adapters

Then run this script to query the server.
"""

import os
import json
import re
import argparse
from typing import List, Dict, Optional
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

class VLLMConfig:
    """Configuration for vLLM inference."""
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = None,  # Will be auto-detected from vLLM
        lora_adapter_name: str = "recursive_reasoner",  # Name used in --lora-modules
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.lora_adapter_name = lora_adapter_name
        self.max_tokens = max_tokens
        self.temperature = temperature


# =============================================================================
# VLLM CLIENT
# =============================================================================

class VLLMClient:
    """Client for vLLM server with LoRA adapter support."""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # Get model name from server if not specified
        if config.model is None:
            models = self.client.models.list()
            self.model = models.data[0].id
            print(f"Using model: {self.model}")
        else:
            self.model = config.model
    
    def generate(
        self, 
        prompt: str, 
        use_lora: bool = True,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """Generate response, optionally using LoRA adapter."""
        
        # vLLM uses model name with adapter suffix for LoRA
        model_name = self.model
        if use_lora and self.config.lora_adapter_name:
            # Check if adapter is available
            # vLLM exposes LoRA adapters as separate model entries
            model_name = self.config.lora_adapter_name
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
        )
        
        return response.choices[0].message.content


# =============================================================================
# RECURSIVE REASONER
# =============================================================================

class RecursiveReasonerVLLM:
    """
    Recursive reasoner using vLLM with LoRA adapters.
    """
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.client = VLLMClient(config)
    
    def parse_json_safe(self, text: str) -> Optional[Dict]:
        """Safely parse JSON from model output."""
        try:
            text = re.sub(r"```(?:json)?", "", text)
            text = re.sub(r"```", "", text)
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
                json_text = re.sub(
                    r'"([^"]*(?:\\"[^"]*)*)"',
                    lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
                    json_text
                )
                return json.loads(json_text)
        except:
            pass
        return None
    
    def plan(self, problem: str, max_width: int = 3) -> Dict:
        """Use planner to analyze problem."""
        prompt = f"""Analyze this problem and create a plan for solving it:

Problem: {problem}

Your task:
1. Determine if this problem should be broken into subproblems or solved directly
2. If decomposing: identify the distinct subproblems (max {max_width})
3. Create a brief plan explaining your approach

DECOMPOSE only if ALL conditions are met:
- The problem EXPLICITLY asks for 2+ UNRELATED things (connected by AND, comma, etc.)
- Each part requires COMPLETELY DIFFERENT expertise
- The parts can be solved independently with zero overlap

KEEP ATOMIC (solve directly) if ANY of these apply:
- The problem is about ONE topic, even if complex
- It asks for steps/process (sequential, not independent)
- It asks about one domain (ethics, math, science, etc.)
- The answer would be a single coherent response

Return ONLY valid JSON:
{{
    "should_decompose": true/false,
    "subproblems": ["subproblem 1", "subproblem 2", ...] or [] if atomic,
    "plan": "Brief description of how to approach this problem"
}}"""
        
        response = self.client.generate(prompt, use_lora=True)
        parsed = self.parse_json_safe(response)
        
        if parsed:
            return {
                'should_decompose': parsed.get('should_decompose', False),
                'subproblems': parsed.get('subproblems', [])[:max_width],
                'plan': parsed.get('plan', 'No plan provided'),
                'raw_response': response
            }
        
        return {
            'should_decompose': False,
            'subproblems': [],
            'plan': 'Direct solution (fallback)',
            'raw_response': response
        }
    
    def solve_subproblem(self, subproblem: str, original_problem: str, plan: str) -> str:
        """Solve a single subproblem."""
        prompt = f"""Solve ONLY this specific task: {subproblem}

Context: This is part of the larger question "{original_problem}"
Plan context: {plan}

⚠️ CRITICAL RULES:
1. Answer ONLY the specific task above - nothing else
2. Do NOT answer other parts of the original question
3. Stay focused on this ONE task
4. Be thorough but concise

RESPONSE FORMAT - Return ONLY valid JSON:
{{"solution": "Your focused answer to the specific task only"}}"""
        
        response = self.client.generate(prompt, use_lora=True)
        parsed = self.parse_json_safe(response)
        
        if parsed:
            return parsed.get('solution', response)
        return response
    
    def solve_atomic(self, problem: str, plan: str) -> str:
        """Solve problem directly."""
        prompt = f"""Solve this problem: {problem}

Approach guidance: {plan}

Provide a thorough, well-structured answer.

RESPONSE FORMAT - Return ONLY valid JSON:
{{"solution": "Your complete answer"}}"""
        
        response = self.client.generate(prompt, use_lora=True)
        parsed = self.parse_json_safe(response)
        
        if parsed:
            return parsed.get('solution', response)
        return response
    
    def combine_solutions(
        self, 
        original_problem: str, 
        subproblems: List[str], 
        solutions: List[str],
        plan: str
    ) -> str:
        """Combine solutions into final answer."""
        solution_text = ""
        for i, (sp, sol) in enumerate(zip(subproblems, solutions)):
            solution_text += f"Subproblem {i+1}: \"{sp}\"\n"
            solution_text += f"Solution {i+1}: {sol}\n\n"
        
        prompt = f"""Synthesize these partial solutions into ONE coherent answer.

Original question: {original_problem}
Original plan: {plan}

Partial solutions:
{solution_text}

SYNTHESIS RULES:
1. REMOVE REDUNDANCY: Include repeated information only ONCE
2. UNIFY: Create a single flowing answer
3. PRIORITIZE: Put the most important information first
4. BE CONCISE: Every sentence should add new information

Return ONLY valid JSON:
{{"combined_solution": "Your synthesized answer here"}}"""
        
        response = self.client.generate(prompt, use_lora=True)
        parsed = self.parse_json_safe(response)
        
        if parsed:
            return parsed.get('combined_solution', response)
        return response
    
    def solve(self, problem: str, max_width: int = 3, verbose: bool = False) -> Dict:
        """
        Main solving function.
        
        Returns dict with:
        - solution: final answer
        - plan: planner's plan
        - is_atomic: whether solved atomically
        - subproblems: list of subproblems (if decomposed)
        - sub_solutions: list of solutions (if decomposed)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Problem: {problem[:80]}{'...' if len(problem) > 80 else ''}")
            print(f"{'='*60}")
        
        # Step 1: Plan
        if verbose:
            print("\n[PLANNER] Analyzing...")
        
        plan_result = self.plan(problem, max_width)
        
        if verbose:
            print(f"  Decision: {'DECOMPOSE' if plan_result['should_decompose'] else 'ATOMIC'}")
            print(f"  Plan: {plan_result['plan']}")
            if plan_result['should_decompose']:
                print(f"  Subproblems: {plan_result['subproblems']}")
        
        # Step 2: Solve
        if not plan_result['should_decompose']:
            if verbose:
                print("\n[WORKER] Solving atomically...")
            
            solution = self.solve_atomic(problem, plan_result['plan'])
            
            return {
                'solution': solution,
                'plan': plan_result['plan'],
                'is_atomic': True,
                'subproblems': [],
                'sub_solutions': []
            }
        
        # Decomposed solving
        subproblems = plan_result['subproblems']
        
        if verbose:
            print(f"\n[WORKER] Solving {len(subproblems)} subproblems...")
        
        sub_solutions = []
        for i, sp in enumerate(subproblems):
            if verbose:
                print(f"  [{i+1}/{len(subproblems)}] {sp[:50]}...")
            sol = self.solve_subproblem(sp, problem, plan_result['plan'])
            sub_solutions.append(sol)
            if verbose:
                print(f"      -> {sol[:60]}...")
        
        # Combine
        if verbose:
            print("\n[WORKER] Combining solutions...")
        
        combined = self.combine_solutions(
            problem, subproblems, sub_solutions, plan_result['plan']
        )
        
        return {
            'solution': combined,
            'plan': plan_result['plan'],
            'is_atomic': False,
            'subproblems': subproblems,
            'sub_solutions': sub_solutions
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Recursive Reasoner with vLLM")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                       help="vLLM server URL")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (auto-detected if not specified)")
    parser.add_argument("--lora-adapter", type=str, default="recursive_reasoner",
                       help="LoRA adapter name as specified in --lora-modules")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    args = parser.parse_args()
    
    config = VLLMConfig(
        base_url=args.base_url,
        model=args.model,
        lora_adapter_name=args.lora_adapter,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    print("="*60)
    print("Recursive Reasoner (vLLM + LoRA)")
    print("="*60)
    print(f"Server: {config.base_url}")
    print(f"LoRA adapter: {config.lora_adapter_name}")
    print("="*60)
    
    try:
        reasoner = RecursiveReasonerVLLM(config)
    except Exception as e:
        print(f"\nError connecting to vLLM server: {e}")
        print("\nMake sure vLLM server is running with LoRA enabled:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model meta-llama/Llama-3.1-8B-Instruct \\")
        print("    --enable-lora \\")
        print("    --lora-modules recursive_reasoner=./grpo_lora_adapters")
        return
    
    while True:
        try:
            user_input = input("\nEnter your question (or 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            result = reasoner.solve(user_input, verbose=True)
            
            print("\n" + "="*60)
            print("FINAL SOLUTION:")
            print("="*60)
            print(result['solution'])
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()

