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

# Replace the main block with a simpler version that can be ignored when imported
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