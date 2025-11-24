"""
Merge LoRA adapter with base model and push to Hugging Face Hub.

Usage:
    python merge_and_push.py --repo-name your-username/model-name
    python merge_and_push.py --repo-name your-username/model-name --adapter-path ./my_adapter
    python merge_and_push.py --repo-name your-username/model-name --model-card ./README.md
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_push(
    base_model_name: str,
    adapter_path: str,
    repo_name: str,
    model_card_path: str = None,
    local_save_path: str = None,
    push_to_hub: bool = True,
    private: bool = False,
):
    """
    Merge LoRA adapter with base model and optionally push to HF Hub.
    
    Args:
        base_model_name: Name of the base model (e.g., meta-llama/Llama-3.1-8B-Instruct)
        adapter_path: Path to the LoRA adapter directory
        repo_name: Hugging Face repo name (e.g., username/model-name)
        model_card_path: Optional path to a custom model card (README.md)
        local_save_path: Optional local path to save merged model
        push_to_hub: Whether to push to Hugging Face Hub
        private: Whether the HF repo should be private
    """
    print("="*60)
    print("Merging LoRA Adapter with Base Model")
    print("="*60)
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Target repo: {repo_name}")
    print("="*60)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load base model
    print("\n[2/5] Loading base model (this may take a while)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load and merge LoRA adapter
    print("\n[3/5] Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge LoRA weights into base model
    print("  Merging weights...")
    model = model.merge_and_unload()
    
    print(f"  Merged model parameters: {model.num_parameters():,}")
    
    # Save locally if requested
    if local_save_path:
        print(f"\n[4/5] Saving merged model locally to {local_save_path}...")
        model.save_pretrained(local_save_path)
        tokenizer.save_pretrained(local_save_path)
        print("  Local save complete!")
    else:
        print("\n[4/5] Skipping local save (no path specified)")
    
    # Push to Hugging Face Hub
    if push_to_hub:
        print(f"\n[5/5] Pushing to Hugging Face Hub: {repo_name}...")
        
        # Push model
        model.push_to_hub(
            repo_name,
            private=private,
            commit_message="Upload merged model",
        )
        
        # Push tokenizer
        tokenizer.push_to_hub(
            repo_name,
            private=private,
            commit_message="Upload tokenizer",
        )
        
        # Push model card if provided
        if model_card_path:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo="README.md",
                repo_id=repo_name,
                repo_type="model",
                commit_message="Add model card",
            )
            print(f"  Model card uploaded from {model_card_path}")
        
        print(f"\n✓ Model pushed successfully!")
        print(f"  View at: https://huggingface.co/{repo_name}")
    else:
        print("\n[5/5] Skipping push to Hub (--no-push specified)")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model and push to HF Hub"
    )
    
    parser.add_argument(
        "--base-model", 
        type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter-path", 
        type=str, 
        default="./adapter",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--repo-name", 
        type=str, 
        required=True,
        help="Hugging Face repo name (e.g., username/model-name)"
    )
    parser.add_argument(
        "--model-card", 
        type=str, 
        default=None,
        help="Optional path to model card (README.md)"
    )
    parser.add_argument(
        "--local-save-path", 
        type=str, 
        default=None,
        help="Optional local path to save merged model"
    )
    parser.add_argument(
        "--no-push", 
        action="store_true",
        help="Don't push to HF Hub (only save locally)"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make the HF repo private"
    )
    
    args = parser.parse_args()
    
    merge_and_push(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        repo_name=args.repo_name,
        model_card_path=args.model_card,
        local_save_path=args.local_save_path,
        push_to_hub=not args.no_push,
        private=args.private,
    )


if __name__ == "__main__":
    main()

