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
        result = process_prompt()
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