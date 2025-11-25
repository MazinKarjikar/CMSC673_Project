#!/usr/bin/env python3
"""
Trained Planner REPL - Uses the GRPO-trained planner from HuggingFace.
Planner: vkaarti/recursive-reasoner-planner-llama3.1-8b (TRAINED)
Worker: meta-llama/Llama-3.1-8B-Instruct (BASE)

Usage:
    # Terminal 1: Start vLLM with trained planner (port 8000)
    python -m vllm.entrypoints.openai.api_server \
        --model vkaarti/recursive-reasoner-planner-llama3.1-8b \
        --port 8000

    # Terminal 2: Start vLLM with base worker (port 8001)
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --port 8001

    # Terminal 3: Run the REPL
    python repl_trained.py
"""

import argparse
import json
import re
import sys
from typing import Optional, Tuple

import requests


class VLLMClient:
    """Simple client for vLLM OpenAI-compatible API."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url.rstrip("/")
    
    def generate(
        self, 
        prompt: str, 
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        system_prompt: str = "You are a helpful math assistant. Answer concisely.",
    ) -> str:
        """Generate completion using chat API (handles Llama 3.1 chat template)."""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def list_models(self) -> list:
        """List available models on the server."""
        response = requests.get(f"{self.base_url}/models", timeout=10)
        response.raise_for_status()
        return [m["id"] for m in response.json()["data"]]


def parse_json(text: str) -> Optional[dict]:
    """Robustly parse JSON from model output."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str.strip())
            except (json.JSONDecodeError, IndexError):
                continue
    
    return None


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def strip_ansi(text: str) -> str:
    return re.sub(r'\033\[[0-9;]*m', '', text)


def wrap_text(text: str, width: int) -> list:
    if len(strip_ansi(text)) <= width:
        return [text]
    
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = f"{current_line} {word}".strip() if current_line else word
        if len(strip_ansi(test_line)) <= width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            if len(strip_ansi(word)) > width:
                while len(strip_ansi(word)) > width:
                    lines.append(word[:width])
                    word = word[width:]
                current_line = word
            else:
                current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines if lines else [""]


def box(title: str, content: str, color: str = Colors.CYAN, width: int = 74) -> str:
    inner_width = width - 4
    
    lines = []
    for line in content.split('\n'):
        wrapped = wrap_text(line, inner_width)
        lines.extend(wrapped)
    
    result = []
    result.append(f"{color}┌{'─' * (width - 2)}┐{Colors.RESET}")
    
    title_display = f" {title} "
    padding_total = width - 2 - len(title_display)
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left
    result.append(f"{color}│{' ' * padding_left}{Colors.BOLD}{title_display}{Colors.RESET}{color}{' ' * padding_right}│{Colors.RESET}")
    
    result.append(f"{color}├{'─' * (width - 2)}┤{Colors.RESET}")
    
    for line in lines:
        visible_len = len(strip_ansi(line))
        padding = inner_width - visible_len
        result.append(f"{color}│{Colors.RESET} {line}{' ' * padding} {color}│{Colors.RESET}")
    
    result.append(f"{color}└{'─' * (width - 2)}┘{Colors.RESET}")
    
    return '\n'.join(result)


def header(text: str, color: str = Colors.BLUE) -> str:
    return f"\n{color}{Colors.BOLD}━━━ {text} ━━━{Colors.RESET}"


def subheader(text: str, color: str = Colors.CYAN) -> str:
    return f"\n{color}{Colors.BOLD}► {text}{Colors.RESET}"


class RecursiveReasoner:
    """Recursive reasoning system with trained planner and base worker."""
    
    def __init__(
        self,
        planner_client: VLLMClient,
        worker_client: VLLMClient,
        planner_model: str,
        worker_model: str,
        verbose: bool = True,
    ):
        self.planner_client = planner_client  # For trained planner
        self.worker_client = worker_client    # For base worker
        self.planner_model = planner_model
        self.worker_model = worker_model
        self.verbose = verbose
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def plan(self, problem: str) -> dict:
        """Use trained planner to decide how to approach the problem."""
        prompt = f"""Analyze this math problem and decide how to solve it:

Problem: {problem}

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
        
        response = self.planner_client.generate(prompt, self.planner_model, max_tokens=512)
        
        result = parse_json(response)
        if result is None:
            return {"should_decompose": False, "subproblems": [], "plan": "Direct solution"}
        
        return result
    
    def solve_subproblem(self, subproblem: str, original_question: str) -> Tuple[str, dict]:
        prompt = f"""Solve this specific part of a math problem:

Original question: {original_question}
Subproblem to solve: {subproblem}

Solve ONLY this subproblem. Show your calculation and give a numerical result.

Return JSON: {{"solution": "your calculation", "result": <number>}}"""
        
        response = self.worker_client.generate(prompt, self.worker_model, max_tokens=256)
        parsed = parse_json(response)
        return response.strip(), parsed
    
    def solve_atomic(self, problem: str) -> Tuple[str, dict]:
        prompt = f"""Solve this math problem step by step:

Problem: {problem}

Show your work clearly and provide the final numerical answer.

Return JSON: {{"solution": "your step-by-step solution", "final_answer": <number>}}"""
        
        response = self.worker_client.generate(prompt, self.worker_model, max_tokens=512)
        parsed = parse_json(response)
        return response.strip(), parsed
    
    def combine(self, problem: str, subproblems: list, solutions: list) -> Tuple[str, dict]:
        solutions_text = ""
        for i, (sp, sol) in enumerate(zip(subproblems, solutions)):
            solutions_text += f"Step {i+1}: {sp}\nResult: {sol}\n\n"
        
        prompt = f"""Combine these partial solutions to answer the original question:

Original question: {problem}

Partial solutions:
{solutions_text}

Use these results to compute the final answer. Show your work.

Return JSON: {{"final_calculation": "how you combined results", "final_answer": <number>}}"""
        
        response = self.worker_client.generate(prompt, self.worker_model, max_tokens=512)
        parsed = parse_json(response)
        return response.strip(), parsed
    
    def solve(self, problem: str) -> str:
        print(header("📝 PROBLEM", Colors.WHITE))
        print(f"\n{problem}\n")
        
        self.log(header("🧠 PHASE 1: PLANNING", Colors.MAGENTA))
        self.log(f"{Colors.DIM}Using planner: {self.planner_model} (TRAINED){Colors.RESET}")
        
        plan = self.plan(problem)
        
        decompose = plan.get('should_decompose', False)
        strategy = plan.get('plan', 'N/A')
        
        if decompose:
            self.log(f"\n{Colors.GREEN}✓ Decision: DECOMPOSE{Colors.RESET}")
        else:
            self.log(f"\n{Colors.YELLOW}✓ Decision: SOLVE DIRECTLY{Colors.RESET}")
        
        self.log(f"{Colors.DIM}Strategy: {strategy}{Colors.RESET}")
        
        if decompose and plan.get("subproblems"):
            subproblems = plan["subproblems"]
            
            subproblems_text = "\n".join([
                f"{i+1}. {sp}" for i, sp in enumerate(subproblems)
            ])
            self.log("\n" + box("SUBPROBLEMS", subproblems_text, Colors.CYAN))
            
            self.log(header("⚙️  PHASE 2: SOLVING SUBPROBLEMS", Colors.BLUE))
            self.log(f"{Colors.DIM}Using worker: {self.worker_model}{Colors.RESET}")
            
            raw_solutions = []
            parsed_results = []
            
            for i, sp in enumerate(subproblems, 1):
                self.log(subheader(f"Subproblem {i}/{len(subproblems)}", Colors.YELLOW))
                self.log(f"{Colors.BOLD}{sp}{Colors.RESET}\n")
                
                raw_sol, parsed = self.solve_subproblem(sp, problem)
                raw_solutions.append(raw_sol)
                parsed_results.append(parsed)
                
                if parsed and "result" in parsed:
                    display = f"Solution: {parsed.get('solution', 'N/A')}\nResult: {parsed['result']}"
                else:
                    display = raw_sol
                self.log(box(f"Solution {i}", display, Colors.GREEN))
            
            self.log(header("🔗 PHASE 3: COMBINING SOLUTIONS", Colors.CYAN))
            
            combine_preview = "\n\n".join([
                f"[{i+1}] {sp}\n    → Result: {self._extract_result(parsed, raw)}"
                for i, (sp, parsed, raw) in enumerate(zip(subproblems, parsed_results, raw_solutions))
            ])
            self.log("\n" + box("INPUTS TO COMBINE", combine_preview, Colors.MAGENTA))
            
            final_raw, final_parsed = self.combine(problem, subproblems, raw_solutions)
            
            if final_parsed and "final_answer" in final_parsed:
                final_display = f"Calculation: {final_parsed.get('final_calculation', 'N/A')}\n\n{Colors.BOLD}Final Answer: {final_parsed['final_answer']}{Colors.RESET}"
            else:
                final_display = final_raw
        else:
            self.log(header("⚙️  PHASE 2: DIRECT SOLUTION", Colors.BLUE))
            self.log(f"{Colors.DIM}Using worker: {self.worker_model}{Colors.RESET}\n")
            
            final_raw, final_parsed = self.solve_atomic(problem)
            
            if final_parsed and "final_answer" in final_parsed:
                final_display = f"Solution: {final_parsed.get('solution', 'N/A')}\n\n{Colors.BOLD}Final Answer: {final_parsed['final_answer']}{Colors.RESET}"
            else:
                final_display = final_raw
        
        print(header("✅ FINAL ANSWER", Colors.GREEN))
        print("\n" + box("RESULT", final_display, Colors.GREEN))
        
        return final_raw
    
    def _extract_result(self, parsed: Optional[dict], raw: str) -> str:
        if parsed:
            if "result" in parsed:
                return str(parsed["result"])
            if "final_answer" in parsed:
                return str(parsed["final_answer"])
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', raw)
        if numbers:
            return numbers[-1]
        return "N/A"


def check_server(client: VLLMClient) -> bool:
    try:
        models = client.list_models()
        print(f"{Colors.GREEN}✓ Connected to vLLM server{Colors.RESET}")
        print(f"  {Colors.DIM}Available models: {models}{Colors.RESET}")
        return True
    except requests.exceptions.ConnectionError:
        return False


def print_help():
    print(f"""
{Colors.CYAN}{Colors.BOLD}Commands:{Colors.RESET}
  {Colors.YELLOW}/help{Colors.RESET}      Show this help
  {Colors.YELLOW}/models{Colors.RESET}    List available models
  {Colors.YELLOW}/verbose{Colors.RESET}   Toggle verbose output
  {Colors.YELLOW}/planner{Colors.RESET}   Show planner model
  {Colors.YELLOW}/worker{Colors.RESET}    Show worker model
  {Colors.YELLOW}/quit{Colors.RESET}      Exit

{Colors.DIM}Type a math problem below to solve it:{Colors.RESET}""")


def repl(reasoner: RecursiveReasoner, client: VLLMClient):
    print(f"""
{Colors.GREEN}{Colors.BOLD}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  🧠  RECURSIVE REASONER (TRAINED)                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛{Colors.RESET}

{Colors.GREEN}Planner:{Colors.RESET}  {reasoner.planner_model}
          {Colors.DIM}(GRPO-trained for problem decomposition){Colors.RESET}
{Colors.BLUE}Worker:{Colors.RESET}   {reasoner.worker_model}

{Colors.DIM}Type {Colors.YELLOW}/help{Colors.RESET}{Colors.DIM} for commands, or enter a problem to solve.{Colors.RESET}
""")
    
    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/models":
                try:
                    models = client.list_models()
                    print(f"Available models: {models}")
                except Exception as e:
                    print(f"Error: {e}")
            elif cmd == "/verbose":
                reasoner.verbose = not reasoner.verbose
                print(f"Verbose mode: {'ON' if reasoner.verbose else 'OFF'}")
            elif cmd == "/planner":
                print(f"Planner model: {reasoner.planner_model}")
            elif cmd == "/worker":
                print(f"Worker model: {reasoner.worker_model}")
            else:
                print(f"Unknown command: {cmd}")
                print_help()
        else:
            try:
                reasoner.solve(user_input)
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Error communicating with server: {e}")
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Trained Planner REPL - Uses GRPO-trained planner from HuggingFace"
    )
    parser.add_argument(
        "--planner-api",
        type=str,
        default="http://localhost:8000/v1",
        help="Planner vLLM API base URL (port 8000)"
    )
    parser.add_argument(
        "--worker-api",
        type=str,
        default="http://localhost:8001/v1",
        help="Worker vLLM API base URL (port 8001)"
    )
    parser.add_argument(
        "--planner-model",
        type=str,
        default="vkaarti/recursive-reasoner-planner-llama3.1-8b",
        help="Trained planner model"
    )
    parser.add_argument(
        "--worker-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Worker model (base)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create separate clients for planner and worker
    planner_client = VLLMClient(args.planner_api)
    worker_client = VLLMClient(args.worker_api)
    
    # Check planner server
    print(f"{Colors.MAGENTA}Checking planner server...{Colors.RESET}")
    if not check_server(planner_client):
        print(f"\n❌ Cannot connect to planner server at {args.planner_api}")
        print("\n   Start the trained planner:")
        print("   python -m vllm.entrypoints.openai.api_server \\")
        print("       --model vkaarti/recursive-reasoner-planner-llama3.1-8b \\")
        print("       --port 8000")
        sys.exit(1)
    
    # Check worker server
    print(f"\n{Colors.BLUE}Checking worker server...{Colors.RESET}")
    if not check_server(worker_client):
        print(f"\n❌ Cannot connect to worker server at {args.worker_api}")
        print("\n   Start the base worker:")
        print("   python -m vllm.entrypoints.openai.api_server \\")
        print("       --model meta-llama/Llama-3.1-8B-Instruct \\")
        print("       --port 8001")
        sys.exit(1)
    
    reasoner = RecursiveReasoner(
        planner_client=planner_client,
        worker_client=worker_client,
        planner_model=args.planner_model,
        worker_model=args.worker_model,
        verbose=not args.quiet,
    )
    
    repl(reasoner, planner_client)


if __name__ == "__main__":
    main()

