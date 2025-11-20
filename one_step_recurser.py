import concurrent.futures
import json
import re
from openai import OpenAI

# vllm setup
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# For now, planner and worker use the same model
# Later these can be different models trained with GRPO + LoRA
planner_model = model
worker_model = model


def parse_json_response(text):
    """
    Cleans up a text string to extract a JSON object.
    Removes markdown code fences and extracts the JSON substring.
    Handles multi-line strings by converting them to escaped newlines.
    Raises a JSONDecodeError if parsing fails.
    """
    # Remove markdown formatting
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    # Extract JSON substring if extra text exists
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    json_text = json_match.group(1) if json_match else text.strip()
    
    # Try parsing as-is first
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    
    # Fix multi-line strings: find content between quotes and escape newlines
    fixed_text = re.sub(
        r'"([^"]*(?:\\"[^"]*)*)"',
        lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
        json_text
    )
    
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass
    
    # Last resort: extract values directly using regex
    for key in ['solution', 'combined_solution', 'plan', 'subproblems']:
        if key == 'subproblems':
            array_match = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', json_text, re.DOTALL)
            if array_match:
                items = re.findall(r'"([^"]*)"', array_match.group(1))
                return {key: items}
        else:
            pattern = rf'"{key}"\s*:\s*"(.*?)"(?:\s*[,}}])'
            match = re.search(pattern, json_text, re.DOTALL)
            if match:
                value = match.group(1).replace('\\"', '"')
                return {key: value}
    
    raise json.JSONDecodeError("Could not parse JSON", json_text, 0)


# =============================================================================
# PLANNER MODEL FUNCTIONS
# =============================================================================

def plan_problem(client, model, problem, max_width=3, verbose=False):
    """
    PLANNER MODEL: Analyzes the problem and generates subproblems with a plan.
    
    Returns:
        dict with keys:
            - 'should_decompose': bool - whether the problem needs decomposition
            - 'subproblems': list[str] - the subproblems to solve (empty if atomic)
            - 'plan': str - the plan for how to approach the problem
    """
    messages = [
        {"role": "user", "content": f"""Analyze this problem and create a plan for solving it:

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
}}"""}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6
        )
        content = response.choices[0].message.content
        if content is None:
            print("Warning: Received None content from planner model")
            return {
                'should_decompose': False,
                'subproblems': [],
                'plan': 'Direct solution approach'
            }
        
        try:
            result = parse_json_response(content)
            should_decompose = result.get('should_decompose', False)
            subproblems = result.get('subproblems', [])
            plan = result.get('plan', 'No plan provided')
            
            # Validate and clean subproblems
            subproblems = [s for s in subproblems if s and isinstance(s, str)]
            subproblems = subproblems[:max_width]
            
            # If should_decompose but no valid subproblems, treat as atomic
            if should_decompose and not subproblems:
                should_decompose = False
            
            if verbose:
                print(f"Planner decision: {'DECOMPOSE' if should_decompose else 'ATOMIC'}")
                if should_decompose:
                    print(f"Subproblems: {subproblems}")
                print(f"Plan: {plan}")
            
            return {
                'should_decompose': should_decompose,
                'subproblems': subproblems,
                'plan': plan
            }
            
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON from planner. Raw response:", content)
            return {
                'should_decompose': False,
                'subproblems': [],
                'plan': 'Direct solution approach (fallback)'
            }
            
    except Exception as e:
        print(f"Error in plan_problem: {e}")
        return {
            'should_decompose': False,
            'subproblems': [],
            'plan': f'Error in planning: {str(e)}'
        }


# =============================================================================
# WORKER MODEL FUNCTIONS
# =============================================================================

def solve_subproblem(client, model, subproblem, original_prompt, plan, verbose=False):
    """
    WORKER MODEL: Solves a single subproblem.
    
    Args:
        subproblem: The specific subproblem to solve
        original_prompt: The original user question (for context)
        plan: The planner's plan (for guidance)
    
    Returns:
        str: The solution to the subproblem
    """
    messages = [
        {"role": "user", "content": f"""Solve ONLY this specific task: {subproblem}

Context: This is part of the larger question "{original_prompt}"
Plan context: {plan}

⚠️ CRITICAL RULES:
1. Answer ONLY the specific task above - nothing else
2. Do NOT answer other parts of the original question
3. Stay focused on this ONE task
4. Be thorough but concise

RESPONSE FORMAT - Return ONLY valid JSON:
{{"solution": "Your focused answer to the specific task only"}}"""}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6
        )
        content = response.choices[0].message.content
        if content is None:
            print("Warning: Received None content from worker model in solve_subproblem")
            return "Unable to get a response for this subproblem."
        
        try:
            result = parse_json_response(content)
            return result.get("solution", content)
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON from worker. Raw response:", content)
            return f"JSON parse error. Raw response: {content}"
            
    except Exception as e:
        print(f"Error in solve_subproblem: {e}")
        return f"Error solving subproblem: {str(e)}"


def solve_atomic(client, model, problem, plan, verbose=False):
    """
    WORKER MODEL: Solves an atomic (non-decomposed) problem directly.
    
    Args:
        problem: The problem to solve
        plan: The planner's plan (for guidance)
    
    Returns:
        str: The solution
    """
    messages = [
        {"role": "user", "content": f"""Solve this problem: {problem}

Approach guidance: {plan}

Provide a thorough, well-structured answer.

RESPONSE FORMAT - Return ONLY valid JSON:
{{"solution": "Your complete answer"}}"""}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6
        )
        content = response.choices[0].message.content
        if content is None:
            print("Warning: Received None content from worker model in solve_atomic")
            return "Unable to get a response."
        
        try:
            result = parse_json_response(content)
            return result.get("solution", content)
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON in solve_atomic. Raw response:", content)
            return content.strip()
            
    except Exception as e:
        print(f"Error in solve_atomic: {e}")
        return f"Error: {str(e)}"


def combine_solutions(client, model, original_problem, subproblems, sub_solutions, plan, verbose=False):
    """
    WORKER MODEL: Combines solutions from subproblems into a unified answer.
    
    Args:
        original_problem: The original user question
        subproblems: List of subproblems that were solved
        sub_solutions: List of solutions to those subproblems
        plan: The planner's plan (for guidance on synthesis)
    
    Returns:
        str: The combined solution
    """
    # Filter out failed solutions
    valid_pairs = [
        (subproblem, solution) 
        for subproblem, solution in zip(subproblems, sub_solutions)
        if isinstance(solution, str) and not solution.startswith("Error") and not solution.startswith("JSON parse error")
    ]
    
    if not valid_pairs:
        return solve_atomic(client, model, original_problem, plan, verbose)
    
    subproblem_solutions = ""
    for i, (subproblem, solution) in enumerate(valid_pairs):
        subproblem_solutions += f"Subproblem {i+1}: \"{subproblem}\"\n"
        subproblem_solutions += f"Solution {i+1}: {solution}\n\n"
    
    messages = [
        {"role": "user", "content": f"""Synthesize these partial solutions into ONE coherent answer.

Original question: {original_problem}
Original plan: {plan}

Partial solutions:
{subproblem_solutions}

SYNTHESIS RULES:
1. REMOVE REDUNDANCY: Include repeated information only ONCE
2. UNIFY: Create a single flowing answer, not separate solutions pasted together
3. PRIORITIZE: Put the most important information first
4. BE CONCISE: Every sentence should add new information
5. NATURAL LANGUAGE: Write as if directly answering the original question

Return ONLY valid JSON:
{{"combined_solution": "Your synthesized answer here"}}"""}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6
        )
        content = response.choices[0].message.content
        if content is None:
            return "Unable to combine solutions due to model error."
        
        try:
            result = parse_json_response(content)
            return result.get("combined_solution", content)
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON in combine_solutions. Raw response:", content)
            combined_match = re.search(r'"combined_solution":\s*"(.*?)"', content)
            if combined_match:
                return combined_match.group(1)
            return content.strip()
            
    except Exception as e:
        print(f"Error in combine_solutions: {e}")
        return f"Error combining solutions: {str(e)}"


# =============================================================================
# MAIN ONE-STEP RECURSION FLOW
# =============================================================================

def one_step_solve(client, planner_model, worker_model, problem, max_width=3, verbose=False):
    """
    Main one-step recursion function.
    
    Flow:
    1. PLANNER: Analyze problem, decide to decompose or not, generate plan
    2. WORKER: If atomic, solve directly. If decomposed, solve each subproblem then combine.
    
    Args:
        client: OpenAI client
        planner_model: Model ID for the planner
        worker_model: Model ID for the worker
        problem: The user's question/problem
        max_width: Maximum number of subproblems
        verbose: Print detailed progress
    
    Returns:
        dict with keys:
            - 'solution': The final solution
            - 'plan': The planner's plan
            - 'is_atomic': Whether the problem was solved atomically
            - 'subproblems': List of subproblems (empty if atomic)
            - 'sub_solutions': List of subproblem solutions (empty if atomic)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {problem[:100]}{'...' if len(problem) > 100 else ''}")
        print(f"{'='*60}")
    
    # Step 1: PLANNER decides how to approach the problem
    if verbose:
        print("\n[PLANNER] Analyzing problem...")
    
    plan_result = plan_problem(client, planner_model, problem, max_width, verbose)
    
    should_decompose = plan_result['should_decompose']
    subproblems = plan_result['subproblems']
    plan = plan_result['plan']
    
    # Step 2: WORKER solves the problem
    if not should_decompose:
        # Atomic case: solve directly
        if verbose:
            print("\n[WORKER] Solving atomically...")
        
        solution = solve_atomic(client, worker_model, problem, plan, verbose)
        
        return {
            'solution': solution,
            'plan': plan,
            'is_atomic': True,
            'subproblems': [],
            'sub_solutions': []
        }
    
    # Decomposed case: solve each subproblem in parallel, then combine
    if verbose:
        print(f"\n[WORKER] Solving {len(subproblems)} subproblems in parallel...")
    
    sub_solutions = [None] * len(subproblems)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subproblems)) as executor:
        future_to_index = {
            executor.submit(
                solve_subproblem, 
                client, 
                worker_model, 
                subprob, 
                problem, 
                plan, 
                verbose
            ): idx
            for idx, subprob in enumerate(subproblems)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                sub_solutions[idx] = future.result()
                if verbose:
                    print(f"  Completed subproblem {idx+1}/{len(subproblems)}")
            except Exception as e:
                if verbose:
                    print(f"  Subproblem {idx+1} failed: {e}")
                sub_solutions[idx] = f"Error solving subproblem: {str(e)}"
    
    # Handle any None results
    for idx, sol in enumerate(sub_solutions):
        if sol is None:
            sub_solutions[idx] = "Subproblem was not processed."
    
    # Step 3: WORKER combines the solutions
    if verbose:
        print("\n[WORKER] Combining solutions...")
    
    combined_solution = combine_solutions(
        client, worker_model, problem, subproblems, sub_solutions, plan, verbose
    )
    
    return {
        'solution': combined_solution,
        'plan': plan,
        'is_atomic': False,
        'subproblems': subproblems,
        'sub_solutions': sub_solutions
    }


def get_response(prompt, max_width=3, verbose=False):
    """
    Process a prompt using one-step recursion and return the solution.
    
    Args:
        prompt (str): The user's question or prompt
        max_width (int, optional): Maximum number of subproblems. Defaults to 3.
        verbose (bool, optional): Print detailed progress. Defaults to False.
    
    Returns:
        str: The final solution
        dict: The full result object if verbose=True
    """
    try:
        result = one_step_solve(
            client, 
            planner_model, 
            worker_model, 
            prompt, 
            max_width, 
            verbose
        )
        
        if verbose:
            print("\n" + "="*60)
            print("RESULTS SUMMARY")
            print("="*60)
            print(f"Atomic: {result['is_atomic']}")
            print(f"Plan: {result['plan']}")
            if not result['is_atomic']:
                print(f"Subproblems solved: {len(result['subproblems'])}")
                for i, (sp, sol) in enumerate(zip(result['subproblems'], result['sub_solutions'])):
                    print(f"  {i+1}. {sp[:50]}{'...' if len(sp) > 50 else ''}")
                    sol_preview = sol[:100] if isinstance(sol, str) else str(sol)[:100]
                    print(f"     -> {sol_preview}{'...' if len(str(sol)) > 100 else ''}")
            return result['solution'], result
        
        return result['solution']
        
    except Exception as e:
        if verbose:
            print(f"Error processing prompt: {e}")
        error_message = f"An error occurred while processing your request: {str(e)}"
        if verbose:
            return error_message, {'solution': error_message, 'error': str(e)}
        return error_message


def print_results(result):
    """Pretty print the results."""
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    print(f"\nPlan: {result['plan']}")
    print(f"Approach: {'Atomic (direct solve)' if result['is_atomic'] else 'Decomposed'}")
    
    if not result['is_atomic']:
        print(f"\nSubproblems ({len(result['subproblems'])}):")
        for i, (sp, sol) in enumerate(zip(result['subproblems'], result['sub_solutions'])):
            print(f"\n  [{i+1}] {sp}")
            print(f"      Solution: {sol}")
    
    print(f"\n{'='*60}")
    print("FINAL SOLUTION:")
    print("="*60)
    print(result['solution'])


if __name__ == "__main__":
    print("One-Step Recurser")
    print("="*60)
    print(f"Planner model: {planner_model}")
    print(f"Worker model: {worker_model}")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nEnter your question (or 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            verbose_mode = True
            if verbose_mode:
                response, result_data = get_response(user_input, verbose=True)
                print_results(result_data)
            else:
                response = get_response(user_input, verbose=False)
                print("\n=== FINAL RESPONSE ===")
                print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
