import json
import re
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

# We use the 32B Qwen Distill
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

try:
    model = client.models.list().data[0].id
    print(f"Connected to model: {model}")
except Exception as e:
    print(f"Error connecting to vLLM: {e}")
    exit(1)

# ================================
# Config
# ================================
OUTPUT_PATH = "synthetic_planning_dataset_local.jsonl"
MAX_EXAMPLES = 5 # for testing stuffs
TEMPERATURE = 0.0

# ================================
# Robust JSON Parsing
# ================================
def parse_json_response(text: str):
    """Extracts and parses JSON from LLM output, handling markdown and errors."""
    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    
    # Find the first JSON object using regex
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    
    json_str = match.group(1)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: simple repair for common trailing comma issues
        # (LLMs often leave a comma after the last item in a list)
        fixed_text = re.sub(r",\s*\}", "}", json_str)
        fixed_text = re.sub(r",\s*\]", "]", fixed_text)
        return json.loads(fixed_text)

# ================================
# IMPROVED PLANNER PROMPT
# ================================
PLANNER_PROMPT = """
You are the Lead Architect for a Recursive AI Solver.
Your goal is to convert a solved "Reasoning Trace" into a "Search Plan" for a blind agent.

The Agent does NOT know the answer. The Agent can only:
1. Search the web / Retrieve knowledge
2. Perform calculations
3. Analyze logic

Your task: Break the `Reasoning Trace` down into the **Dependency Graph of Questions** that had to be answered to get there.

────────────────────────────────
RULES FOR SUB-PROBLEMS
────────────────────────────────
1. **INTERROGATIVE or IMPERATIVE:** Sub-problems must be things to *do* or *find*, never facts to *read*.
   - BAD (Factual Leak): "The medulla controls breathing."
   - GOOD (Plan): "Identify the specific brain structure responsible for breathing control."

2. **INDEPENDENCE (Important):** Try to make sub-problems independent so they can be solved in parallel.
   - If Step B needs Step A's answer, try to frame Step B as a conditional or general query.

3. **ATOMICITY:**
   - If the Problem is simple fact retrieval (e.g. "What is a cell?"), set `"decompose": false`.
   - If the Problem requires combining 2+ pieces of info, set `"decompose": true`.

────────────────────────────────
EXAMPLE 1 (Biology - Concept Linking)
────────────────────────────────
Problem: How does the nervous system regulate heart rate?
Reasoning Trace: The sympathetic system accelerates HR via norepinephrine. The parasympathetic slows it via acetylcholine. They work in opposition.
Plan:
{{
  "decompose": true,
  "subproblems": [
    "Identify the two main divisions of the autonomic nervous system involved in heart regulation.",
    "Determine the neurotransmitters released by the sympathetic nervous system and their effect on heart rate.",
    "Determine the neurotransmitters released by the parasympathetic nervous system and their effect on heart rate.",
    "Synthesize how these two systems interact to maintain balance."
  ],
  "max_depth_hint": 1
}}

────────────────────────────────
EXAMPLE 2 (Math - Step-by-Step)
────────────────────────────────
Problem: A store sells 3 packs of gum for $2. How much for 15 packs?
Reasoning Trace: 15 packs is 5 groups of 3. So 5 * $2 = $10.
Plan:
{{
  "decompose": true,
  "subproblems": [
    "Determine the number of '3-pack groups' needed to make 15 packs.",
    "Calculate the total cost by multiplying the number of groups by the price per group."
  ],
  "max_depth_hint": 1
}}

────────────────────────────────
YOUR INPUT
────────────────────────────────
Problem:
{problem}

Reasoning Trace:
{reasoning}

OUTPUT JSON ONLY.
"""

# ================================
# Guardrails (Safety Filter) - We will improve this in the future, placeholder
# ================================
def is_safe_plan(plan):
    if not plan.get("decompose", False):
        return True 
    subproblems = plan.get("subproblems", [])
    dump = json.dumps(subproblems).lower()
    
    # AI cannot physically interact with the world.
    physical_triggers = [
        "mix", "sample", "patient", "microscope", "dilute", "load", "weigh", 
        "palpate", "surgery", "go to", "measure the"
    ]
    
    for word in physical_triggers:
        if f" {word} " in dump or dump.startswith(word):
            return False
            
    return True

# ================================
# LLM Generation Function
# ================================
def generate_plan(problem: str, reasoning: str):
    # Format the prompt using the fixed template
    prompt_content = PLANNER_PROMPT.format(
        problem=problem,
        reasoning=reasoning
    )

    messages = [
        {"role": "user", "content": prompt_content}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=1024
        )
        content = response.choices[0].message.content
        return parse_json_response(content)
        
    except Exception as e:
        return None

# ================================
# Main Loop
# ================================
print("Loading dataset...")
dataset = load_dataset("allenai/big-reasoning-traces", "DeepSeek", split="train[:5%]")

written = 0
skipped_unsafe = 0
skipped_errors = 0

print(f"Starting generation (Max: {MAX_EXAMPLES})...")

with open(OUTPUT_PATH, "w") as f:
    for ex in tqdm(dataset):
        if written >= MAX_EXAMPLES:
            break

        problem = ex.get("prompt")
        reasoning = ex.get("response")
        
        # Validation: Ensure prompt/response exist and aren't too short
        if not problem or not reasoning: continue
        if len(problem.split()) < 4: continue

        # Generate Plan
        plan = generate_plan(problem, reasoning)
        
        if not plan:
            skipped_errors += 1
            continue

        # Apply Safety Filter
        if not is_safe_plan(plan):
            skipped_unsafe += 1
            continue

        # Write valid record
        record = {
            "problem": problem,
            "plan": plan
        }
        f.write(json.dumps(record) + "\n")
        written += 1

print(f"\nGeneration Complete.")
print(f"Written: {written}")
print(f"Skipped (Unsafe/Physical): {skipped_unsafe}")
print(f"Skipped (Generation Errors): {skipped_errors}")
print(f"Output saved to: {OUTPUT_PATH}")