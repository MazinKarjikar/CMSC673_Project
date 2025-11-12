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

def solve_problem(client, model, problem, original_prompt, depth=0, max_depth=5, max_width=3, verbose=False):
    if verbose:
        print(f"Processing problem at depth {depth}: {problem if len(problem) < 100 else problem[:100] + '...'}")

    if depth >= max_depth or not should_decompose():
        solution = solve_atomic_problem()
        return {
            'solution': solution,
            'atomic': True,
            'depth': depth,
            'original_problem': problem,
            'subproblems': [],
            'sub_solutions': [],
            'sub_results': []
        }

    subproblems = break_down_problem()
    if not subproblems:
        solution = solve_atomic_problem()
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
    combined_solution = combine_solutions()
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

if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nEnter your question (or 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print("\nAn error occurred:", e)