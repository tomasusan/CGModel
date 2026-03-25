import json
import torch
from config import cfg
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from human_eval.data import read_problems

# Load base model path from configuration
base_model_path = cfg.get("model", 'teacher_model')
# Path to LoRA weights for code alignment fine-tuning
lora_path = f"lora/{base_model_path}_code_alignment"

# Initialize tokenizer for the base model
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True  # Allow loading custom model code
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    device_map="auto",  # Automatically distribute across available devices
    trust_remote_code=True
)

# Load and merge LoRA weights
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()  # Merge LoRA weights into base model
model.eval()  # Set to evaluation mode

# Load HumanEval problems
problems = read_problems()

print(f"{len(problems)} problems loaded")

results = []

from tqdm import tqdm

# Number of generations per problem (for pass@k calculation)
repeat = 1
print(f"Generating results for pass@{repeat} for {base_model_path}")

# Iterate through all problems with progress bar
for task_id, problem in tqdm(problems.items(), total=len(problems)):

    def build_prompt(in_prompt):
        """
        Build a formatted prompt for code generation.

        Args:
            in_prompt: Raw problem prompt from HumanEval

        Returns:
            Formatted prompt with user/assistant markers
        """
        return f"<|user|>\nsolve problem below, write code only:\n{in_prompt}\n<|assistant|>\n"


    def extract_code(in_text):
        """
        Extract pure code from generated text by removing special tokens.

        Args:
            in_text: Generated text that may contain special tokens

        Returns:
            Cleaned code string
        """
        # Remove assistant marker if present
        if "<|assistant|>" in in_text:
            in_text = in_text.split("<|assistant|>")[-1]
        # Remove user marker if present
        if "<|user|>" in in_text:
            in_text = in_text.split("<|user|>")[0]
        # Remove think tags if present (used in some reasoning models)
        if "</think>" in in_text:
            in_text = in_text.replace("</think>", "")
        if "<think>" in in_text:
            in_text = in_text.replace("<think>", "")
        return in_text.strip()


    # Build prompt and tokenize
    prompt = build_prompt(problem["prompt"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate code without gradient computation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # Maximum tokens to generate
            do_sample=False  # Greedy decoding for reproducibility
        )

    # Decode generated tokens
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract completion part (remove prompt)
    completion = text[len(prompt):]
    # Clean up special tokens
    completion = extract_code(completion)

    # Store result
    results.append({
        "task_id": task_id,
        "completion": completion
    })

from pathlib import Path

# Define output path for generated samples
path = Path(f"test/human_eval/{base_model_path}_samples_pass@{repeat}.jsonl")
# Create parent directory if it doesn't exist
path.parent.mkdir(parents=True, exist_ok=True)

# Write results to JSONL file (one JSON object per line)
with open(path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("Generation complete")