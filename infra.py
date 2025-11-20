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
    # This handles cases where the model puts actual newlines inside JSON string values
    def fix_multiline_strings(match):
        content = match.group(1)
        # Escape actual newlines within the string
        content = content.replace('\n', '\\n').replace('\r', '\\r')
        # Also escape any unescaped quotes
        content = re.sub(r'(?<!\\)"', '\\"', content)
        return f'"{content}"'
    
    # Match string values (content between quotes, handling the key-value pattern)
    # This regex finds ": " followed by a quoted string value
    fixed_text = re.sub(
        r'"([^"]*(?:\\"[^"]*)*)"',
        lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
        json_text
    )
    
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass
    
    # Last resort: extract the value directly using regex
    # Look for "solution": "..." or "combined_solution": "..." etc.
    for key in ['solution', 'combined_solution', 'decision', 'subproblems']:
        pattern = rf'"{key}"\s*:\s*"(.*?)"(?:\s*[,}}])'
        match = re.search(pattern, json_text, re.DOTALL)
        if match:
            value = match.group(1).replace('\\"', '"')
            if key == 'subproblems':
                # Handle array case
                array_match = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', json_text, re.DOTALL)
                if array_match:
                    items = re.findall(r'"([^"]*)"', array_match.group(1))
                    return {key: items}
            return {key: value}
    
    # If all else fails, raise the original error
    raise json.JSONDecodeError("Could not parse JSON", json_text, 0)

def solve_atomic_problem(client, model, problem, original_prompt, verbose=False):
    messages = [
        {"role": "user", "content": f"""Solve ONLY this specific task: {problem}

Context (for reference only): This relates to the broader question "{original_prompt}"

⚠️ CRITICAL RULES:
1. Answer ONLY the specific task above - nothing else
2. Do NOT answer other parts of the original question
3. Stay focused on this ONE task
4. If the task is about math, give ONLY the math answer
5. If the task is about ethics, discuss ONLY ethics
6. Do NOT combine topics

RESPONSE FORMAT - Return ONLY valid JSON (no newlines inside the string value):
{{"solution": "Your focused answer to the specific task only"}}

Keep your answer on a single line or use \\n for line breaks within the JSON string."""}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6
        )
        content = response.choices[0].message.content
        if content is None:
            print("Warning: Received None content from the model in solve_atomic_problem")
            return "Unable to get a response for this problem."
        try:
            result = parse_json_response(content)
            return result.get("solution", content)
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON from model response in solve_atomic_problem")
            print("Raw response:", content)
            return f"JSON parse error. Raw response: {content}"
    except Exception as e:
        print("Error in solve_atomic_problem:", e)
        if 'response' in locals() and hasattr(response, 'choices') and response.choices:
            print("Raw model output:", response.choices[0].message.content)
        return f"Error solving atomic problem. Error: {e}"

def should_decompose(client, model, problem, depth, max_depth, verbose=False):
    if depth >= max_depth:
        return False

    messages = [
        {"role": "user", "content": f"""Analyze whether this problem requires decomposition: {problem}

DECOMPOSE only if ALL of these conditions are met:
1. The problem EXPLICITLY asks for 2+ UNRELATED things (often connected by "AND", "also", or comma)
2. Each part requires COMPLETELY DIFFERENT expertise (e.g., calculus vs philosophy)
3. You could give each part to a different specialist with zero overlap
4. The problem cannot be answered coherently as one response

KEEP ATOMIC (do NOT decompose) if ANY of these apply:
- The problem is about ONE topic, even if complex
- It asks for steps/process (these are sequential, not independent)
- It asks about one domain (ethics, math, science, etc.)
- Subproblems would share ANY vocabulary or concepts
- It's already a focused subproblem from a larger question
- The answer would be a single coherent paragraph or list

ATOMIC EXAMPLES (never decompose these patterns):
- "What should I do if X?" → ATOMIC (single scenario)
- "Explain/Describe X" → ATOMIC (single topic)
- "What are the ethical considerations of X?" → ATOMIC (single analysis)
- "Discuss biases in X" → ATOMIC (single topic)
- "Find the derivative of X" → ATOMIC (single calculation)
- "What are pros and cons of X?" → ATOMIC (single analysis)

DECOMPOSE EXAMPLES (must have explicit AND connecting unrelated topics):
- "Find derivative of f(x)=3x² AND discuss AI ethics in medicine" → DECOMPOSE (math AND ethics)
- "Write a haiku AND solve 2+2" → DECOMPOSE (creative AND math)

Respond with ONLY valid JSON:
{{"decision": "DECOMPOSE"}} or {{"decision": "ATOMIC"}}"""}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6
        )
        content = response.choices[0].message.content
        if content is None:
            print("Warning: Received None content from the model in should_decompose")
            return False
        try:
            result = parse_json_response(content)
            decision = result.get("decision", "").strip().upper()
            return decision == "DECOMPOSE"
        except json.JSONDecodeError:
            print("JSON parse error in should_decompose. Raw response:", content)
            return "DECOMPOSE" in content.strip().upper()
    except Exception as e:
        print("Error in should_decompose:", e)
        return False

def break_down_problem(client, model, problem, original_prompt, depth=0, max_width=3, verbose=False):
    messages = [
        {"role": "user", "content": f"""Extract the EXPLICITLY SEPARATE parts of this problem:

Problem: {problem}
Original question: {original_prompt}

RULES:
1. ONLY decompose if the problem contains EXPLICIT separate requests (connected by AND, comma, etc.)
2. Each subproblem must be a DIRECT QUOTE or close paraphrase from the original
3. Do NOT invent new subproblems or expand the scope
4. Do NOT break a single topic into aspects/perspectives/steps

VALID decomposition (explicit AND connecting unrelated topics):
- "Calculate 2+2 AND write a poem" → ["Calculate 2+2", "Write a poem"]
- "Explain photosynthesis AND discuss economic policy" → ["Explain photosynthesis", "Discuss economic policy"]

INVALID decomposition (single topic, no explicit AND):
- "Discuss AI ethics" → [] (single topic - don't split into "biases", "fairness", etc.)
- "What should I do after a crash?" → [] (single scenario - don't split into steps)
- "Explain quantum computing" → [] (single topic - don't split into aspects)

If there are NOT 2+ explicitly separate requests, return EMPTY: {{"subproblems": []}}

Return ONLY valid JSON on a single line:
{{"subproblems": ["exact part 1 from problem", "exact part 2 from problem"]}}"""}
    ]

    try:
        breakdown = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6
        )
        breakdown_text = breakdown.choices[0].message.content
        if breakdown_text is None:
            print("Warning: Received None content from the model in break_down_problem")
            return [f"Solve the problem directly: {problem}"]
        try:
            result = parse_json_response(breakdown_text)
            subproblems = result.get("subproblems", [])
            subproblems = [s for s in subproblems if s and isinstance(s, str)]
            # Enforce max_width limit
            subproblems = subproblems[:max_width]
            if subproblems:
                print(f"Extracted {len(subproblems)} subproblems at depth {depth}.")
                return subproblems
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON in break_down_problem, falling back to text parsing")
            print("Raw response:", breakdown_text)
            subproblems = []
            lines = breakdown_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                match = re.match(r'^(\d+\.\s*|\*\s*|\-\s*)(.*)', line)
                if match and match.group(2).strip():
                    subproblems.append(match.group(2).strip())
            if not subproblems and ',' in breakdown_text:
                subproblems = [s.strip() for s in breakdown_text.split(',')]
            # Enforce max_width limit
            subproblems = subproblems[:max_width]
            return subproblems if subproblems else [f"Solve the problem directly: {problem}"]
    except Exception as e:
        print("Error in break_down_problem:", e)
        return [f"Solve the problem directly: {problem}"]

def combine_solutions(client, model, original_problem, subproblems, sub_solutions, original_prompt, verbose=False):
    # Filter out failed solutions
    valid_pairs = [(subproblem, solution) for subproblem, solution in zip(subproblems, sub_solutions)
                  if isinstance(solution, str) and not solution.startswith("Error") and not solution.startswith("JSON parse error")]

    if not valid_pairs:
        return solve_atomic_problem(client, model, original_problem, original_prompt, verbose)

    subproblem_solutions = ""
    for i, (subproblem, solution) in enumerate(valid_pairs):
        subproblem_solutions += f"Subproblem {i+1}: \"{subproblem}\"\n"
        subproblem_solutions += f"Solution {i+1}: {solution}\n\n"

    messages = [
        {"role": "user", "content": f"""Synthesize these partial solutions into ONE coherent answer.

Original question: {original_prompt}
Current problem: {original_problem}

Partial solutions:
{subproblem_solutions}

SYNTHESIS RULES:
1. REMOVE REDUNDANCY: If multiple solutions mention the same advice, include it only ONCE
2. UNIFY: Create a single flowing answer, not a list of separate solutions pasted together
3. PRIORITIZE: Put the most important/urgent information first
4. BE CONCISE: Every sentence should add new information
5. NATURAL LANGUAGE: Write as if you're directly answering the original question

BAD synthesis (redundant):
"Check for injuries. Also check if anyone is hurt. Make sure to look for injuries."

GOOD synthesis (unified):
"First, check yourself and others for injuries."

Return ONLY valid JSON:
{{"combined_solution": "Your synthesized answer here"}}"""}
    ]
    try:
        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6
        )
        content = final_response.choices[0].message.content
        if content is None:
            return "Unable to combine solutions due to model error."
        try:
            result = parse_json_response(content)
            return result.get("combined_solution", content)
        except json.JSONDecodeError:
            # Fallback extraction using regex if JSON parsing fails.
            print("Warning: Failed to parse JSON in combine_solutions. Raw response:", content)
            combined_match = re.search(r'"combined_solution":\s*"(.*?)"', content)
            if combined_match:
                return combined_match.group(1)
            return content.strip()
    except Exception as e:
        print("Error in combine_solutions:", str(e))
        return f"Error combining solutions: {str(e)}"

def solve_problem(client, model, problem, original_prompt, depth=0, max_depth=5, max_width=3, verbose=False):
    if verbose:
        print(f"Processing problem at depth {depth}: {problem if len(problem) < 100 else problem[:100] + '...'}")

    if depth >= max_depth or not should_decompose(client, model, problem, depth, max_depth):
        solution = solve_atomic_problem(client, model, problem, original_prompt, verbose)
        return {
            'solution': solution,
            'atomic': True,
            'depth': depth,
            'original_problem': problem,
            'subproblems': [],
            'sub_solutions': [],
            'sub_results': []
        }

    subproblems = break_down_problem(client, model, problem, original_prompt, depth, max_width, verbose)
    if not subproblems:
        solution = solve_atomic_problem(client, model, problem, original_prompt, verbose)
        return {
            'solution': solution,
            'atomic': True,
            'depth': depth,
            'original_problem': problem,
            'subproblems': [],
            'sub_solutions': [],
            'sub_results': []
        }

    sub_results = [None] * len(subproblems)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1024) as executor:
        future_to_index = {
            executor.submit(solve_problem, client, model, subprob, original_prompt, depth + 1, max_depth, max_width, verbose): idx
            for idx, subprob in enumerate(subproblems)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                sub_results[idx] = future.result()
                if verbose:
                    print(f"Completed subproblem {idx+1}/{len(subproblems)} at depth {depth+1}")
            except Exception as e:
                if verbose:
                    print(f"Subproblem {idx+1} at depth {depth+1} failed: {e}")
                sub_results[idx] = {
                    'solution': f"Error solving this part: {str(e)}",
                    'atomic': True,
                    'depth': depth + 1,
                    'original_problem': subproblems[idx],
                    'subproblems': [],
                    'sub_solutions': [],
                    'sub_results': []
                }

    if None in sub_results:
        missing_indices = [i for i, r in enumerate(sub_results) if r is None]
        if verbose:
            print("Warning: Some subproblems were not processed:", missing_indices)
        for idx in missing_indices:
            sub_results[idx] = {
                'solution': "This subproblem was not processed.",
                'atomic': True,
                'depth': depth + 1,
                'original_problem': subproblems[idx],
                'subproblems': [],
                'sub_solutions': [],
                'sub_results': []
            }

    sub_solutions = [result['solution'] for result in sub_results]
    if verbose:
        print(f"Combining {len(sub_solutions)} solutions at depth {depth}")
    combined_solution = combine_solutions(client, model, problem, subproblems, sub_solutions, original_prompt, verbose)
    return {
        'solution': combined_solution,
        'atomic': False,
        'depth': depth,
        'original_problem': problem,
        'subproblems': subproblems,
        'sub_solutions': sub_solutions,
        'sub_results': sub_results
    }

def process_prompt(client, model, prompt, max_depth=4, max_width=3, verbose=False):
    return solve_problem(client, model, prompt, prompt, 0, max_depth, max_width, verbose)

def get_response(prompt, max_depth=3, max_width=3, verbose=False):
    """
    Process a prompt and return the final solution.

    Args:
        prompt (str): The user's question or prompt
        max_depth (int, optional): Maximum recursion depth for problem decomposition. Defaults to 3.
        max_width (int, optional): Maximum number of subproblems at each level. Defaults to 3.
        verbose (bool, optional): If True, prints detailed progress and returns detailed results. Defaults to False.

    Returns:
        str: The final solution to the prompt
        dict: The full result object if verbose=True, otherwise None
    """
    try:
        result = process_prompt(client, model, prompt, max_depth, max_width, verbose)
        if verbose:
            print("\n=== DETAILED RESULTS ===")
            print_results(result)
            return result['solution'], result
        return result['solution']
    except Exception as e:
        if verbose:
            print(f"Error processing prompt: {e}")
        error_message = f"An error occurred while processing your request: {str(e)}"
        return error_message

def print_results(result, indent=0):
    indent_str = "  " * indent

    def safe_truncate(s, length=100):
        s = str(s) if not isinstance(s, str) else s
        return (s[:length] + '...') if len(s) > length else s

    if result.get('atomic', False):
        print(f"{indent_str}[ATOMIC] {safe_truncate(result.get('original_problem', 'Problem'))}")
        print(f"{indent_str}Solution: {safe_truncate(result['solution'])}")
    else:
        print(f"{indent_str}Problem: {safe_truncate(result.get('original_problem', 'Root problem'))}")
        print(f"{indent_str}Final solution: {safe_truncate(result['solution'])}")
        if result.get('sub_results'):
            # Count successful vs failed subproblems
            successful = sum(1 for r in result['sub_results'] if not r['solution'].startswith("Error"))
            total = len(result['sub_results'])

            print(f"{indent_str}Subproblems ({successful}/{total} successful):")
            for i, sub_result in enumerate(result['sub_results']):
                status = "[OK]" if not sub_result['solution'].startswith("Error") else "[FAILED]"
                print(f"{indent_str}  #{i+1} {status}:")
                print_results(sub_result, indent + 2)

if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nEnter your question (or 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            verbose_mode = True  # Set to False to disable verbose mode
            if verbose_mode:
                response, result_data = get_response(user_input, verbose=True)
                print("\n=== DETAILED RESULTS ===")
                print_results(result_data)
            else:
                response = get_response(user_input, verbose=False)

            print("\n=== FINAL RESPONSE ===")
            print(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print("\nAn error occurred:", e)